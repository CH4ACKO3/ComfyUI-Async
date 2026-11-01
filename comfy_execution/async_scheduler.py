"""Shared device workers for the process-isolated graph executors."""

import concurrent.futures
import json
import os
import queue
import secrets
import socket
import threading
import time
from multiprocessing.connection import Listener

from comfy_execution.async_execution import ObjectRef, StaleObjectError, Worker, WorkerError, map_values


class DeviceQueue:
    def __init__(self, scheduler, device):
        self.scheduler = scheduler
        self.generation = secrets.token_hex(16)
        self.queue = queue.Queue()
        self.pending = 0
        self.remaining = 0.0
        self.durations = {}
        self.stats = {}
        self.owners = set()
        self.closed = False
        self.worker = Worker({
            "role": "node", "device": "default" if device == scheduler.default_device else device,
            "worker_id": device, "generation": self.generation,
            "lock_directory": scheduler.directory,
        }, scheduler.forward)
        threading.Thread(target=self.run, name=f"device-{device}", daemon=True).start()

    def submit(self, command, payload):
        future = concurrent.futures.Future()
        if self.closed:
            future.set_exception(WorkerError("GPU worker was stopped"))
            return future
        if command == "call":
            self.owners.add(payload["owner"])
        elif command == "release":
            self.owners.discard(payload["owner"])
        kind = payload.get("class_type", command)
        estimate = self.durations.get(kind, 1.0)
        self.pending += 1
        self.remaining += estimate
        self.queue.put((command, payload, kind, estimate, future))
        return future

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            command, payload, kind, estimate, future = item
            start = time.monotonic()
            try:
                if self.closed:
                    raise WorkerError("GPU worker was stopped")
                if command == "call" and self.scheduler.cancelled(payload["prompt_id"]):
                    raise WorkerError("Execution interrupted")
                response = self.worker.submit(command, payload)
            except Exception as exc:
                response = concurrent.futures.Future()
                response.set_exception(exc)
            response.add_done_callback(lambda response, item=item, start=start: self.finish(response, item, start))

    def finish(self, response, item, start):
        command, payload, kind, estimate, future = item
        try:
            result = response.result()
            self.stats = result["stats"]
            if command == "call" and self.scheduler.cancelled(payload["prompt_id"]):
                raise WorkerError("Execution interrupted")
            future.set_result(result["value"])
        except Exception as exc:
            if not self.closed and self.worker.process is not None and (self.worker.closed or self.worker.connection is None or self.worker.process.poll() is not None):
                self.close()
                future.set_exception(StaleObjectError(self.worker.config["worker_id"]))
            else:
                future.set_exception(exc)
        finally:
            with self.scheduler.mutex:
                elapsed = time.monotonic() - start
                self.durations[kind] = 0.75 * self.durations.get(kind, elapsed) + 0.25 * elapsed
                self.pending -= 1
                self.remaining = max(0.0, self.remaining - estimate)

    def close(self):
        self.closed = True
        self.worker.close()
        self.queue.put(None)


class DeviceScheduler:
    def __init__(self, directory, default_device, send_event):
        self.directory = directory
        self.default_device = default_device
        self.send_event = send_event
        self.workers = {}
        self.routes = {}
        self.subscribers = {}
        self.mutex = threading.RLock()
        self.authkey = secrets.token_bytes(32)
        self.listener = Listener(("127.0.0.1", 0), backlog=socket.SOMAXCONN, authkey=self.authkey)
        self.config = {"broker_address": self.listener.address, "broker_authkey": self.authkey.hex()}
        threading.Thread(target=self.accept, name="gpu-scheduler", daemon=True).start()

    def cancelled(self, prompt_id):
        return os.path.exists(os.path.join(self.directory, prompt_id + ".cancel"))

    def forward(self, event, data, sid=None):
        if isinstance(data, dict):
            prompt_id = data.get("prompt_id")
        elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
            prompt_id = data[1].get("prompt_id")
        else:
            prompt_id = None
        with self.mutex:
            subscribers = self.subscribers.get(prompt_id)
            routes = [(event_id, self.routes.get(event_id)) for event_id in subscribers] if subscribers is not None else [(prompt_id, self.routes.get(prompt_id))]
        for event_id, route in routes:
            if route is not None:
                if isinstance(data, dict):
                    routed = dict(data, prompt_id=event_id)
                elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
                    routed = (data[0], dict(data[1], prompt_id=event_id))
                else:
                    routed = data
                route(event, routed, sid)
            elif prompt_id is None:
                self.send_event(event, data, sid)

    def accept(self):
        while True:
            try:
                connection = self.listener.accept()
            except (OSError, EOFError):
                return
            threading.Thread(target=self.serve, args=(connection,), daemon=True).start()

    def serve(self, connection):
        send_lock = threading.Lock()

        def respond(request_id, command, payload):
            try:
                result = (request_id, "result", self.request(command, payload))
            except StaleObjectError as exc:
                result = (request_id, "stale", exc.device)
            except Exception as exc:
                result = (request_id, "error", str(exc))
            try:
                with send_lock:
                    connection.send(result)
            except (OSError, EOFError):
                pass

        with connection:
            try:
                while True:
                    request_id, command, payload = connection.recv()
                    if command in ("subscribe", "unsubscribe", "cancel"):
                        respond(request_id, command, payload)
                    else:
                        threading.Thread(target=respond, args=(request_id, command, payload), daemon=True).start()
            except (OSError, EOFError):
                pass

    def device(self, name):
        worker = self.workers.get(name)
        if worker is None or worker.closed:
            worker = self.workers[name] = DeviceQueue(self, name)
        return worker

    def request(self, command, payload):
        with self.mutex:
            if command == "subscribe":
                self.subscribers.setdefault(payload["call_id"], set()).add(payload["prompt_id"])
                return None
            if command == "unsubscribe":
                subscribers = self.subscribers.get(payload["call_id"], set())
                subscribers.discard(payload["prompt_id"])
                if not subscribers:
                    self.subscribers.pop(payload["call_id"], None)
                return None
            if command == "cancel":
                with open(os.path.join(self.directory, payload["prompt_id"] + ".cancel"), "w"):
                    pass
                return None
            if command in ("trim", "release", "retire"):
                devices = {ref.worker for ref in payload["objects"]} if command == "retire" else self.workers.keys()
                futures = [worker.submit(command, payload) for device, worker in self.workers.items() if device in devices and not worker.closed and payload["owner"] in worker.owners]
            elif command == "export":
                ref = payload["ref"]
                worker = self.workers[ref.worker]
                if worker.closed or worker.generation != ref.generation:
                    raise StaleObjectError(ref.worker)
                futures = [worker.submit(command, payload)]
            else:
                if self.cancelled(payload["prompt_id"]):
                    raise WorkerError("Execution interrupted")
                owners = set()

                def collect(value):
                    if isinstance(value, ObjectRef):
                        worker = self.workers.get(value.worker)
                        if worker is None or worker.closed or worker.generation != value.generation:
                            raise StaleObjectError(value.worker)
                        if not value.transferable:
                            owners.add(value.worker)
                    return value

                map_values(payload["inputs"], collect)
                devices = payload["devices"]
                if len(owners) > 1:
                    raise WorkerError("Models/custom objects belong to different GPUs. Put their loaders and consumers in the same GPU group.")
                if owners:
                    device = owners.pop()
                    if devices and device not in devices:
                        raise WorkerError(f"Node {payload['node_id']} is assigned to {devices}, but its model/custom object belongs to {device}. Put the loader in the same GPU group, or use a separate loader for each GPU.")
                else:
                    candidates = devices or [self.default_device]
                    recipe = json.dumps([payload["class_type"], payload["inputs"]], sort_keys=True, default=str)

                    def cost(name):
                        worker = self.workers.get(name)
                        if worker is None:
                            return 2.0
                        warm = recipe in worker.stats.get("model_keys", [])
                        return worker.remaining + (0.0 if warm else 2.0)

                    device = min(candidates, key=cost)
                worker = self.device(device)
                worker.pending += 1
                worker.remaining += 1.0
        if command == "call":
            try:
                transfers = {}

                def transfer(value):
                    if isinstance(value, ObjectRef) and value.worker != device:
                        if value not in transfers:
                            transfers[value] = self.request("export", {"ref": value})
                        return transfers[value]
                    return value

                payload = dict(payload, inputs=map_values(payload["inputs"], transfer))
                with self.mutex:
                    if self.cancelled(payload["prompt_id"]):
                        raise WorkerError("Execution interrupted")
                    futures = [worker.submit(command, payload)]
            finally:
                with self.mutex:
                    worker.pending -= 1
                    worker.remaining = max(0.0, worker.remaining - 1.0)
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except WorkerError:
                if command != "release":
                    raise
        return results[0] if command not in ("trim", "release", "retire") else None

    def snapshot(self):
        with self.mutex:
            return {device: {"pid": worker.worker.process.pid if worker.worker.process else None,
                             "pending": worker.pending, "generation": worker.generation, **worker.stats}
                    for device, worker in self.workers.items() if not worker.closed}

    def clear(self):
        with self.mutex:
            workers = list(self.workers.values())
            self.workers.clear()
        for worker in workers:
            worker.close()

    def close(self):
        self.listener.close()
        self.clear()
