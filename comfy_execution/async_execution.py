"""Process-isolated node execution, with the native executor owning the graph."""

import asyncio
import atexit
import concurrent.futures
import copy
import contextvars
import json
import logging
import os
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener

from comfy_execution.graph_utils import is_link

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


class StaleObjectError(WorkerError):
    def __init__(self, device):
        self.device = device
        super().__init__(f"GPU worker {device} restarted; refresh the affected nodes")


class GraphRevisionError(ValueError):
    pass


def map_values(value, visit):
    if type(value) is dict:
        return {k: map_values(v, visit) for k, v in value.items()}
    if type(value) is list:
        return [map_values(v, visit) for v in value]
    if type(value) is tuple:
        return tuple(map_values(v, visit) for v in value)
    return visit(value)


def graph_key(extra_data):
    return extra_data.get("client_id", ""), extra_data["graph_event"]["graph_id"]


class Worker:
    def __init__(self, config, send_event):
        self.config = config
        self.send_event = send_event
        self.process = None
        self.connection = None
        self.mutex = threading.Lock()
        self.lifecycle = threading.Lock()
        self.closed = False
        self.pending = {}
        self.pending_mutex = threading.Lock()
        self.request_index = 0

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

    def submit(self, command, payload):
        with self.mutex:
            if self.closed:
                raise WorkerError("Execution interrupted")
            if self.process is None:
                self.start()
                threading.Thread(target=self.read_responses, daemon=True).start()
            self.request_index += 1
            request_id = self.request_index
            future = concurrent.futures.Future()
            with self.pending_mutex:
                self.pending[request_id] = future
            try:
                self.connection.send((request_id, command, payload))
            except (BrokenPipeError, OSError) as exc:
                with self.pending_mutex:
                    self.pending.pop(request_id, None)
                raise WorkerError("Execution worker stopped") from exc
            return future

    def read_responses(self):
        try:
            while True:
                kind, payload = self.connection.recv()
                if kind == "event":
                    self.send_event(*payload)
                    continue
                request_id, response = payload
                with self.pending_mutex:
                    future = self.pending.pop(request_id)
                if kind == "error":
                    future.set_exception(WorkerError(response))
                else:
                    future.set_result(response)
        except (EOFError, OSError):
            self.closed = True
            with self.pending_mutex:
                pending, self.pending = self.pending, {}
            for future in pending.values():
                future.set_exception(WorkerError("Execution worker stopped; see the server log"))

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


class BrokerChannel:
    def __init__(self, config):
        self.connection = Client(tuple(config["broker_address"]), authkey=bytes.fromhex(config["broker_authkey"]))
        self.outgoing = queue.Queue()
        self.pending = {}
        self.mutex = threading.Lock()
        self.index = 0
        self.closed = False
        threading.Thread(target=self.write, daemon=True).start()
        threading.Thread(target=self.read, daemon=True).start()

    def submit(self, command, payload):
        with self.mutex:
            if self.closed:
                raise WorkerError("Graph scheduler connection closed")
            self.index += 1
            future = concurrent.futures.Future()
            self.pending[self.index] = future
            self.outgoing.put((self.index, command, payload))
        return future

    def fail(self):
        with self.mutex:
            self.closed = True
            pending, self.pending = self.pending, {}
        for future in pending.values():
            if future.set_running_or_notify_cancel():
                future.set_exception(WorkerError("Graph scheduler connection closed"))
        self.outgoing.put(None)

    def write(self):
        try:
            while (message := self.outgoing.get()) is not None:
                self.connection.send(message)
        except (EOFError, OSError):
            self.fail()

    def read(self):
        try:
            while True:
                request_id, kind, value = self.connection.recv()
                with self.mutex:
                    future = self.pending.pop(request_id, None)
                if future is not None and future.set_running_or_notify_cancel():
                    if kind == "stale":
                        future.set_exception(StaleObjectError(value))
                    elif kind == "error":
                        future.set_exception(WorkerError(value))
                    else:
                        future.set_result(value)
        except (EOFError, OSError):
            self.fail()


class Runtime:
    def __init__(self, config, server, channel, graph, interrupted):
        self.channel = channel
        self.graph = graph
        self.interrupted = interrupted
        self.node_keys = {}
        self.config = config
        self.server = server
        self.owner = config["owner"]
        self.refs = weakref.WeakValueDictionary()
        self.dynprompt = None
        self.prompt_id = None
        self.extra_data = {}
        self.tasks = set()

    def intern(self, value):
        if isinstance(value, ObjectRef):
            value = self.refs.setdefault((value.worker, value.generation, value.key), value)
        return value

    async def call(self, node_id, index, func, inputs, v3_data):
        root_id = self.dynprompt.get_real_node_id(node_id)
        payload = {
            "owner": self.owner,
            "graph_key": graph_key(self.extra_data),
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
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            try:
                result = await self.graph.call(self, node_id, index, func, payload)
            except StaleObjectError as exc:
                self.graph.invalidate_device(exc.device)
                raise
            except (WorkerError, asyncio.CancelledError):
                if self.prompt_id and os.path.exists(os.path.join(self.config["lock_directory"], self.prompt_id + ".cancel")):
                    from comfy.model_management import InterruptProcessingException
                    raise InterruptProcessingException()
                raise
            return map_values(result, self.intern)
        finally:
            self.tasks.discard(task)


class ResidentGraphExecutor:
    def __init__(self, server, queue, asset_manager, default_device):
        self.server = server
        self.queue = queue
        self.asset_manager = asset_manager
        self.directory = tempfile.TemporaryDirectory(prefix="comfy-gpu-")
        self.running = {}
        self.worker = None
        self.cancelled = set()
        self.clear_requested = False
        self.revisions = {}
        self.frames = {}
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
            if self.worker is not None:
                self.worker.close()
                self.worker = None
            self.scheduler.clear()

    def shutdown(self):
        self.stopping.set()
        self.interrupt()
        if self.worker is not None:
            self.worker.close()
        self.scheduler.close()

    def start(self):
        threading.Thread(target=self.dispatch, name="graph-inputs", daemon=True).start()

    def register_graph(self, key, prompt):
        with self.queue.mutex:
            if key in self.frames:
                raise ValueError("Graph already registered")
            self.frames[key] = copy.deepcopy(prompt)
            for node in self.frames[key].values():
                node.setdefault("_meta", {})["input_revision"] = 0
            self.revisions[key] = 0

    def prepare_event(self, key, base_revision, changes, refresh):
        if type(base_revision) is not int or base_revision < 0:
            raise ValueError("base_revision must be a nonnegative integer")
        if not isinstance(changes, dict) or not isinstance(refresh, list) or any(not isinstance(node_id, str) for node_id in refresh):
            raise ValueError("changes must map node IDs to definitions; refresh must list node IDs")
        if not changes and not refresh:
            raise ValueError("An event requires changed or refreshed nodes")
        with self.queue.mutex:
            if key not in self.frames:
                raise KeyError("Register the graph before sending events")
            if base_revision != self.revisions[key]:
                raise GraphRevisionError("Graph revision changed; reload its state before submitting this event")
            prompt = copy.deepcopy(self.frames[key])
        revision = base_revision + 1
        changed = set(changes) | set(refresh)
        for node_id, node in changes.items():
            if node is None:
                if node_id not in prompt:
                    raise ValueError(f"Unknown node {node_id}")
                del prompt[node_id]
            else:
                prompt[node_id] = copy.deepcopy(node)
        for node_id in refresh:
            if node_id not in prompt:
                raise ValueError(f"Unknown node {node_id}")
        for node_id in changed & prompt.keys():
            prompt[node_id].setdefault("_meta", {})["input_revision"] = revision
        children = {node_id: set() for node_id in changed | prompt.keys()}
        for node_id, node in prompt.items():
            for value in node["inputs"].values():
                if is_link(value):
                    children.setdefault(value[0], set()).add(node_id)
        affected = set(changed)
        pending = list(changed)
        while pending:
            for node_id in children[pending.pop()] - affected:
                affected.add(node_id)
                pending.append(node_id)
        affected.intersection_update(prompt)
        targets = [node_id for node_id in prompt if node_id in affected and not children[node_id]]
        return prompt, revision, sorted(changed), affected, targets

    def enqueue(self, item, base_revision):
        key = graph_key(item[3])
        with self.queue.mutex:
            if base_revision != self.revisions.get(key):
                raise GraphRevisionError("Graph changed while validating this event; resubmit it")
            if item[1] in self.queue.history or any(queued[1] == item[1] for queued in self.queue.queue + list(self.queue.currently_running.values())):
                raise ValueError("event_id has already been submitted")
            self.frames[key] = copy.deepcopy(item[2])
            self.revisions[key] = item[3]["graph_event"]["revision"]
            self.queue.put(item)

    def forward(self, item, event, data, sid=None):
        from comfy_execution.asset_enrichment import register_executed_outputs
        from protocol import BinaryEventTypes

        metadata = item[3].get("workflow_metadata", {})
        metadata = {**metadata, "graph_event": item[3]["graph_event"]}
        if isinstance(data, dict) and "prompt_id" in data:
            data = {**metadata, **data}
            if event == "executed":
                data["output"] = register_executed_outputs(data["output"], item[1], self.asset_manager)
        elif event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA and isinstance(data, tuple):
            data = (data[0], {**metadata, **data[1]})
        self.server.send_sync(event, data, sid)

    def dispatch(self):
        while not self.stopping.is_set():
            flags = self.queue.get_flags()
            if flags.get("free_memory") or flags.get("unload_models"):
                self.clear()
            queued = self.queue.get(timeout=0.5)
            if queued is None:
                continue
            item, item_id = queued
            with self.mutex:
                if self.stopping.is_set():
                    return
                if self.worker is None or self.worker.closed:
                    if self.worker is not None:
                        self.worker.close()
                        self.scheduler.clear()
                    self.worker = Worker({"role": "graph", "lock_directory": self.directory.name, **self.scheduler.config}, self.scheduler.forward)
                worker = self.worker
                self.running[item[1]] = worker
                self.scheduler.routes[item[1]] = lambda event, data, sid=None, item=item: self.forward(item, event, data, sid)
            self.asset_manager.pause_background_scan()
            try:
                future = worker.submit("event", {"item": item, "sockets_metadata": self.server.sockets_metadata})
            except Exception as exc:
                future = concurrent.futures.Future()
                future.set_exception(exc)
            future.add_done_callback(lambda response, item=item, item_id=item_id: self.complete(response, item, item_id))

    def complete(self, response, item, item_id):
        import execution
        from comfy_execution.asset_enrichment import register_executed_outputs

        prompt_id, client_id = item[1], item[3].get("client_id")
        try:
            history, success, messages = response.result()
            history["outputs"] = {node_id: register_executed_outputs(output, prompt_id, self.asset_manager) for node_id, output in history["outputs"].items()}
        except Exception as exc:
            logging.exception("Graph input %s failed", prompt_id)
            event = "execution_interrupted" if prompt_id in self.cancelled else "execution_error"
            details = {"prompt_id": prompt_id, "node_id": None, "node_type": "", "executed": [], "exception_message": str(exc), "exception_type": type(exc).__name__, "traceback": [], "timestamp": int(time.time() * 1000)}
            self.forward(item, event, details, client_id)
            history, success, messages = {}, False, [(event, details)]
            self.scheduler.request("cancel", {"prompt_id": prompt_id})
        self.queue.task_done(item_id, history, status=execution.PromptQueue.ExecutionStatus(status_str="success" if success else "error", completed=success, messages=messages), process_item=lambda prompt: prompt[:5] + prompt[6:])
        with self.mutex:
            self.running.pop(prompt_id, None)
            self.scheduler.routes.pop(prompt_id, None)
            self.cancelled.discard(prompt_id)
            if self.clear_requested:
                self.clear()
        self.forward(item, "executing", {"node": None, "prompt_id": prompt_id}, client_id)
        self.asset_manager.queue_output_scan()
        if self.queue.get_tasks_remaining() == 0:
            self.asset_manager.resume_background_scan()
