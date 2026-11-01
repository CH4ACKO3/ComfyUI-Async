"""Process-isolated node execution, with the native executor owning the graph."""

import asyncio
import atexit
import contextvars
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
from dataclasses import dataclass
from multiprocessing.connection import Listener

import psutil


active_runtime = contextvars.ContextVar("async_execution_runtime", default=None)


@dataclass(frozen=True)
class ObjectRef:
    worker: str
    key: int
    transferable: bool
    type_name: str


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
        self.workers = {}
        self.locks = {}
        self.pending = {}
        self.refs = weakref.WeakValueDictionary()
        self.dynprompt = None
        self.prompt_id = None
        self.extra_data = {}

    def intern(self, value):
        if isinstance(value, ObjectRef):
            key = (value.worker, value.key)
            value = self.refs.setdefault(key, value)
        return value

    def devices_for(self, node_id):
        root_id = self.dynprompt.get_real_node_id(node_id)
        return self.dynprompt.get_node(root_id).get("_meta", {}).get("gpu_devices", [])

    def select(self, node_id, inputs):
        owners = set()

        def collect(value):
            if isinstance(value, ObjectRef) and not value.transferable:
                owners.add(value.worker)
            return value

        map_values(inputs, collect)
        devices = self.devices_for(node_id)
        if len(owners) > 1:
            raise WorkerError("This node combines model/custom objects from different workers. Put their loaders and consumers in the same GPU group.")
        if owners:
            owner = owners.pop()
            actual = self.config["default_device"] if owner == "default" else owner
            if devices and actual not in devices:
                raise WorkerError(f"Node {node_id} is assigned to {devices}, but its model/custom object belongs to {actual}. Put the loader in the same GPU group, or use a separate loader for each GPU.")
            return owner
        if not devices:
            return "default"
        return min(devices, key=lambda device: self.pending.get(device, 0))

    def worker(self, device):
        if device not in self.workers:
            config = dict(self.config, role="node", device=device)
            self.workers[device] = Worker(config, self.server.send_sync)
            self.locks[device] = asyncio.Lock()
        return self.workers[device]

    async def request(self, device, command, payload):
        worker = self.worker(device)
        async with self.locks[device]:
            return await asyncio.to_thread(worker.request, command, payload)

    async def call(self, node_id, index, func, inputs, v3_data):
        device = self.select(node_id, inputs)
        self.pending[device] = self.pending.get(device, 0) + 1
        try:
            transfers = {}

            def collect(value):
                if isinstance(value, ObjectRef) and value.worker != device:
                    transfers[(value.worker, value.key)] = value
                return value

            map_values(inputs, collect)
            for key, ref in list(transfers.items()):
                if not ref.transferable:
                    raise WorkerError(f"Cannot transfer {ref.type_name} between GPU workers")
                transfers[key] = await self.request(ref.worker, "export", ref.key)

            def transfer(value):
                if isinstance(value, ObjectRef) and value.worker != device:
                    return transfers[(value.worker, value.key)]
                return value

            payload = {
                "prompt_id": self.prompt_id,
                "node_id": node_id,
                "index": index,
                "class_type": self.dynprompt.get_node(node_id)["class_type"],
                "func": func,
                "inputs": map_values(inputs, transfer),
                "v3_data": v3_data,
                "prompt": self.dynprompt.original_prompt,
                "extra_data": self.extra_data,
                "sockets_metadata": self.server.sockets_metadata,
            }
            result = await self.request(device, "call", payload)
            return map_values(result, self.intern)
        finally:
            self.pending[device] -= 1

    def trim(self):
        nodes = [(node_id, self.dynprompt.get_node(node_id)["class_type"]) for node_id in self.dynprompt.all_node_ids()]
        for device, worker in self.workers.items():
            keys = [key for owner, key in self.refs.keys() if owner == device]
            worker.request("trim", {"objects": keys, "nodes": nodes})

    def close(self):
        for worker in self.workers.values():
            worker.close()


class AsyncExecutorPool:
    def __init__(self, server, queue, asset_manager, slots, default_device):
        self.server = server
        self.queue = queue
        self.asset_manager = asset_manager
        self.slots = slots
        self.default_device = default_device
        self.directory = tempfile.TemporaryDirectory(prefix="comfy-gpu-")
        self.running = {}
        self.idle = set()
        self.generation = 0
        self.cancelled = set()
        self.stopping = threading.Event()
        self.mutex = threading.Lock()
        atexit.register(self.shutdown)

    def interrupt(self, prompt_id=None):
        with self.queue.mutex, self.mutex:
            ids = {item[1] for item in self.queue.currently_running.values() if prompt_id is None or item[1] == prompt_id}
            self.cancelled.update(ids)
            workers = [worker for key, worker in self.running.items() if key in ids]
            for worker in workers:
                worker.closed = True
        for worker in workers:
            worker.close()

    def clear(self):
        with self.mutex:
            self.generation += 1
            workers = list(self.idle)
            for worker in workers:
                worker.closed = True
        for worker in workers:
            worker.close()

    def shutdown(self):
        self.stopping.set()
        self.interrupt()
        self.clear()

    def start(self):
        for index in range(self.slots):
            threading.Thread(target=self.run_slot, name=f"prompt-{index}", daemon=True).start()

    def run_slot(self):
        # Import here to avoid the execution -> async_execution -> execution cycle.
        import execution
        from comfy_execution.asset_enrichment import register_executed_outputs
        from protocol import BinaryEventTypes

        worker = None
        while not self.stopping.is_set():
            flags = self.queue.get_flags()
            if flags.get("free_memory") or flags.get("unload_models"):
                self.clear()
            queued = self.queue.get(timeout=0.5)
            if queued is None:
                continue
            item, item_id = queued
            prompt_id = item[1]
            client_id = item[3].get("client_id")

            def forward(event, data, sid=None):
                metadata = item[3].get("workflow_metadata", {})
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
                    worker = Worker({"role": "prompt", "lock_directory": self.directory.name, "default_device": self.default_device}, forward)
                worker.send_event = forward
                generation = self.generation
                self.running[prompt_id] = worker
                if prompt_id in self.cancelled:
                    worker.closed = True
            self.asset_manager.pause_background_scan()
            try:
                result = worker.request("prompt", {"item": item, "sockets_metadata": self.server.sockets_metadata})
                history, success, messages = result
                history["outputs"] = {node_id: register_executed_outputs(output, prompt_id, self.asset_manager) for node_id, output in history["outputs"].items()}
            except Exception as exc:
                logging.exception("Async prompt %s failed", prompt_id)
                event = "execution_interrupted" if worker.closed else "execution_error"
                details = {"prompt_id": prompt_id, "node_id": None, "node_type": "", "executed": [], "exception_message": str(exc), "exception_type": type(exc).__name__, "traceback": [], "timestamp": int(time.time() * 1000)}
                forward(event, details, client_id)
                history, success, messages = {}, False, [(event, details)]
                worker.close()
            finally:
                with self.mutex:
                    self.running.pop(prompt_id, None)
                    self.idle.add(worker)
                    discard = generation != self.generation
                if discard:
                    worker.close()
            self.queue.task_done(item_id, history, status=execution.PromptQueue.ExecutionStatus(status_str="success" if success else "error", completed=success, messages=messages), process_item=lambda prompt: prompt[:5] + prompt[6:])
            with self.mutex:
                self.cancelled.discard(prompt_id)
            forward("executing", {"node": None, "prompt_id": prompt_id}, client_id)
            self.asset_manager.queue_output_scan()
            if self.queue.get_tasks_remaining() == 0:
                self.asset_manager.resume_background_scan()
