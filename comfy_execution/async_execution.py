"""Process-isolated node execution, with the native executor owning the graph."""

import asyncio
import atexit
import contextvars
import heapq
import json
import logging
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener

import psutil


active_runtime = contextvars.ContextVar("async_execution_runtime", default=None)


@dataclass(frozen=True)
class ObjectRef:
    worker: str
    key: int
    transferable: bool
    generation: str


@dataclass
class ObjectData:
    data: bytes


class WorkerError(RuntimeError):
    pass


def map_values(value, visit):
    if type(value) is dict:
        return {k: map_values(v, visit) for k, v in value.items()}
    if type(value) is list:
        return [map_values(v, visit) for v in value]
    if type(value) is tuple:
        return tuple(map_values(v, visit) for v in value)
    return visit(value)


def session_key(extra_data):
    session = extra_data.get("execution_session")
    return (extra_data.get("client_id", ""), session["id"]) if session else None


class Worker:
    def __init__(self, config, send_event):
        self.config = config
        self.send_event = send_event
        self.process = None
        self.connection = None
        self.mutex = threading.Lock()
        self.lifecycle = threading.Lock()
        self.closed = False

    def start(self):
        authkey = secrets.token_bytes(32)
        with Listener(("127.0.0.1", 0), authkey=authkey) as listener:
            # A fresh interpreter must select its device before importing torch.
            with self.lifecycle:
                if self.closed:
                    raise WorkerError("Execution interrupted")
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "comfy_execution.async_worker"],
                    stdin=subprocess.PIPE,
                    cwd=os.path.dirname(os.path.dirname(__file__)),
                )
            config = dict(self.config, address=listener.address, authkey=authkey.hex(), argv=sys.argv[1:])
            self.process.stdin.write(json.dumps(config).encode() + b"\n")
            self.process.stdin.close()
            # Startup failures must not leave a queue slot waiting forever.
            listener._listener._socket.settimeout(1)
            deadline = time.monotonic() + 180
            while not self.closed:
                try:
                    self.connection = listener.accept()
                    break
                except TimeoutError:
                    if self.process.poll() is not None or time.monotonic() > deadline:
                        raise WorkerError("Execution worker failed to start; see the server log")
            if self.closed:
                raise WorkerError("Execution interrupted")
        self.receive()

    def receive(self):
        while True:
            try:
                kind, payload = self.connection.recv()
            except (EOFError, OSError) as exc:
                raise WorkerError("Execution worker stopped; see the server log") from exc
            if kind == "event":
                self.send_event(*payload)
            elif kind == "error":
                raise WorkerError(payload)
            else:
                return payload

    def request(self, command, payload=None):
        with self.mutex:
            if self.closed:
                raise WorkerError("Execution interrupted")
            if self.process is None:
                self.start()
            try:
                self.connection.send((command, payload))
            except (BrokenPipeError, OSError) as exc:
                raise WorkerError("Execution worker stopped") from exc
            return self.receive()

    def close(self):
        with self.lifecycle:
            self.closed = True
        if self.process is not None:
            try:
                parent = psutil.Process(self.process.pid)
                processes = parent.children(recursive=True) + [parent]
                for process in reversed(processes):
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(processes, timeout=3)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            self.process.wait()
        if self.connection is not None:
            self.connection.close()


class Runtime:
    def __init__(self, config, server):
        self.config = config
        self.server = server
        self.owner = config["owner"]
        self.refs = weakref.WeakValueDictionary()
        self.dynprompt = None
        self.prompt_id = None
        self.extra_data = {}

    def intern(self, value):
        if isinstance(value, ObjectRef):
            value = self.refs.setdefault((value.worker, value.generation, value.key), value)
        return value

    def request(self, command, payload):
        with Client(tuple(self.config["broker_address"]), authkey=bytes.fromhex(self.config["broker_authkey"])) as connection:
            connection.send((command, payload))
            kind, result = connection.recv()
        if kind == "error":
            if command == "call" and self.prompt_id and os.path.exists(os.path.join(self.config["lock_directory"], self.prompt_id + ".cancel")):
                # Imported here because model_management imports execution helpers during startup.
                from comfy.model_management import InterruptProcessingException
                raise InterruptProcessingException()
            raise WorkerError(result)
        return result

    async def call(self, node_id, index, func, inputs, v3_data):
        root_id = self.dynprompt.get_real_node_id(node_id)
        payload = {
            "owner": self.owner,
            "session_id": session_key(self.extra_data),
            "prompt_id": self.prompt_id,
            "node_id": node_id,
            "index": index,
            "class_type": self.dynprompt.get_node(node_id)["class_type"],
            "devices": self.dynprompt.get_node(root_id).get("_meta", {}).get("gpu_devices", []),
            "func": func,
            "inputs": inputs,
            "v3_data": v3_data,
            "prompt": self.dynprompt.original_prompt,
            "extra_data": self.extra_data,
            "sockets_metadata": self.server.sockets_metadata,
        }
        result = await asyncio.to_thread(self.request, "call", payload)
        return map_values(result, self.intern)

    def trim(self):
        nodes = [(node_id, self.dynprompt.get_node(node_id)["class_type"]) for node_id in self.dynprompt.all_node_ids()]
        self.request("trim", {"owner": self.owner, "objects": list(self.refs.values()), "nodes": nodes,
                              "session_id": session_key(self.extra_data)})

    def close(self):
        self.request("release", {"owner": self.owner})


class AsyncExecutorPool:
    def __init__(self, server, queue, asset_manager, slots, default_device):
        self.server = server
        self.queue = queue
        self.asset_manager = asset_manager
        self.slots = slots
        self.directory = tempfile.TemporaryDirectory(prefix="comfy-gpu-")
        self.running = {}
        self.idle = set()
        self.cancelled = set()
        self.clear_requested = False
        self.revisions = OrderedDict()
        self.session_slots = {}
        from comfy_execution.async_scheduler import DeviceScheduler
        self.scheduler = DeviceScheduler(self.directory.name, default_device, server.send_sync)
        self.stopping = threading.Event()
        self.mutex = threading.RLock()
        atexit.register(self.shutdown)

    def interrupt(self, prompt_id=None):
        with self.queue.mutex, self.mutex:
            ids = {item[1] for item in self.queue.currently_running.values() if prompt_id is None or item[1] == prompt_id}
            self.cancelled.update(ids)
            for key in ids:
                with open(os.path.join(self.directory.name, key + ".cancel"), "w"):
                    pass

    def clear(self):
        with self.mutex:
            if self.running:
                self.clear_requested = True
                return
            self.clear_requested = False
            for worker in self.idle:
                worker.close()
            self.idle.clear()
            self.scheduler.clear()

    def shutdown(self):
        self.stopping.set()
        self.interrupt()
        with self.mutex:
            workers = list(self.running.values()) + list(self.idle)
        for worker in workers:
            worker.close()
        self.scheduler.close()

    def start(self):
        for index in range(self.slots):
            threading.Thread(target=self.run_slot, args=(index,), name=f"prompt-{index}", daemon=True).start()

    def enqueue(self, item):
        # Queue ownership makes revision checks and replacement atomic with dequeue.
        import execution

        session = item[3].get("execution_session")
        if session is None:
            self.queue.put(item)
            return
        if (not isinstance(session, dict) or not isinstance(session.get("id"), str)
                or not 1 <= len(session["id"]) <= 128 or type(session.get("revision")) is not int
                or session["revision"] < 0 or session.get("policy", "latest") not in ("latest", "fifo")):
            raise ValueError("execution_session requires id, nonnegative integer revision and latest/fifo policy")
        session = item[3]["execution_session"] = {"id": session["id"], "revision": session["revision"], "policy": session.get("policy", "latest")}
        key = session_key(item[3])
        with self.queue.mutex:
            if session["revision"] <= self.revisions.get(key, -1):
                raise ValueError("execution_session revision must increase")
            pending = [queued for queued in self.queue.queue if session_key(queued[3]) == key]
            if session["policy"] == "fifo" and len(pending) >= 64:
                raise OverflowError("Execution session has 64 pending inputs; wait before submitting more")
            if key not in self.revisions and len(self.revisions) >= 256:
                active = {session_key(queued[3]) for queued in self.queue.queue + list(self.queue.currently_running.values())}
                oldest = next((candidate for candidate in self.revisions if candidate not in active), None)
                if oldest is None:
                    raise OverflowError("Too many active execution sessions")
                del self.revisions[oldest]
                del self.session_slots[oldest]
            if key not in self.session_slots:
                assigned = [self.session_slots.get(session_key(queued[3])) for queued in self.queue.queue + list(self.queue.currently_running.values())]
                self.session_slots[key] = min(range(self.slots), key=lambda slot: (assigned.count(slot), list(self.session_slots.values()).count(slot)))
            self.revisions[key] = session["revision"]
            self.revisions.move_to_end(key)
            if session["policy"] == "latest":
                for queued in pending:
                    self.queue.queue.remove(queued)
                heapq.heapify(self.queue.queue)
            self.queue.put(item)
            if session["policy"] == "latest":
                for queued in pending:
                    details = {"prompt_id": queued[1], "execution_session": queued[3]["execution_session"], "replacement_prompt_id": item[1], "timestamp": int(time.time() * 1000)}
                    item_id = self.queue.task_counter
                    self.queue.task_counter += 1
                    self.queue.currently_running[item_id] = queued
                    self.queue.task_done(item_id, {}, execution.PromptQueue.ExecutionStatus("error", False, [("execution_superseded", details)]), process_item=lambda prompt: prompt[:5])
                    self.server.send_sync("execution_superseded", details, item[3].get("client_id"))

    def accepts(self, item, slot):
        key = session_key(item[3])
        if key is None:
            return True
        return self.session_slots[key] == slot

    def run_slot(self, slot):
        # Import here to avoid the execution -> async_execution -> execution cycle.
        import execution
        from comfy_execution.asset_enrichment import register_executed_outputs
        from protocol import BinaryEventTypes

        worker = None
        while not self.stopping.is_set():
            flags = self.queue.get_flags()
            if flags.get("free_memory") or flags.get("unload_models"):
                self.clear()
            queued = self.queue.get(timeout=0.5, predicate=lambda item: self.accepts(item, slot))
            if queued is None:
                continue
            item, item_id = queued
            prompt_id = item[1]
            client_id = item[3].get("client_id")

            def forward(event, data, sid=None):
                metadata = item[3].get("workflow_metadata", {})
                if "execution_session" in item[3]:
                    metadata = {**metadata, "execution_session": item[3]["execution_session"]}
                if isinstance(data, dict) and "prompt_id" in data:
                    data = {**metadata, **data}
                    if event == "executed":
                        data["output"] = register_executed_outputs(data["output"], prompt_id, self.asset_manager)
                elif event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA and isinstance(data, tuple):
                    data = (data[0], {**metadata, **data[1]})
                self.server.send_sync(event, data, sid)

            with self.mutex:
                if self.stopping.is_set():
                    return
                self.idle.discard(worker)
                if worker is None or worker.closed:
                    worker = Worker({"role": "prompt", "owner": secrets.token_hex(16), "lock_directory": self.directory.name, **self.scheduler.config}, forward)
                worker.send_event = forward
                self.running[prompt_id] = worker
                self.scheduler.routes[prompt_id] = forward
            self.asset_manager.pause_background_scan()
            try:
                result = worker.request("prompt", {"item": item, "sockets_metadata": self.server.sockets_metadata})
                history, success, messages = result
                history["outputs"] = {node_id: register_executed_outputs(output, prompt_id, self.asset_manager) for node_id, output in history["outputs"].items()}
            except Exception as exc:
                logging.exception("Async prompt %s failed", prompt_id)
                event = "execution_interrupted" if prompt_id in self.cancelled else "execution_error"
                details = {"prompt_id": prompt_id, "node_id": None, "node_type": "", "executed": [], "exception_message": str(exc), "exception_type": type(exc).__name__, "traceback": [], "timestamp": int(time.time() * 1000)}
                forward(event, details, client_id)
                history, success, messages = {}, False, [(event, details)]
                self.scheduler.request("cancel", {"prompt_id": prompt_id})
                worker.close()
                self.scheduler.request("release", {"owner": worker.config["owner"]})
            finally:
                with self.mutex:
                    self.running.pop(prompt_id, None)
                    self.idle.add(worker)
                    self.scheduler.routes.pop(prompt_id, None)
                    if self.clear_requested:
                        self.clear()
            self.queue.task_done(item_id, history, status=execution.PromptQueue.ExecutionStatus(status_str="success" if success else "error", completed=success, messages=messages), process_item=lambda prompt: prompt[:5] + prompt[6:])
            with self.mutex:
                self.cancelled.discard(prompt_id)
            forward("executing", {"node": None, "prompt_id": prompt_id}, client_id)
            self.asset_manager.queue_output_scan()
            if self.queue.get_tasks_remaining() == 0:
                self.asset_manager.resume_background_scan()
