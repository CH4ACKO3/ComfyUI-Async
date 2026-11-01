"""Private, authenticated local worker. Started by async_execution.Worker."""

import asyncio
import contextvars
import inspect
import json
import os
import pickle
import sys
import threading
import time
import traceback
from collections import OrderedDict
from contextlib import nullcontext
from multiprocessing.connection import Client

import psutil
from filelock import FileLock

from comfy_execution.async_execution import BrokerChannel, ObjectData, ObjectRef, Runtime, active_runtime, map_values, session_key


call_payload = contextvars.ContextVar("worker_call_payload", default=None)


class CallServer:
    def __init__(self, server):
        self.server = server

    def send_sync(self, event, data, sid=None):
        payload = call_payload.get()
        if payload is not None and "call_id" in payload:
            metadata = {"call_id": payload["call_id"], "list_index": payload["index"]}
            if isinstance(data, dict):
                data = {**data, **metadata}
            elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
                data = (data[0], {**data[1], **metadata})
            if sid is None:
                sid = self.client_id
        self.server.send_sync(event, data, sid)

    def __getattr__(self, name):
        payload = call_payload.get()
        if payload is not None:
            fields = {
                "client_id": payload["extra_data"].get("client_id"),
                "last_prompt_id": payload["prompt_id"], "last_node_id": payload["node_id"],
                "workflow_metadata": payload["extra_data"].get("workflow_metadata", {}),
                "sockets_metadata": payload["sockets_metadata"],
            }
            if name in fields:
                return fields[name]
        return getattr(self.server, name)

    def __setattr__(self, name, value):
        payload = call_payload.get()
        if payload is not None:
            fields = {"last_prompt_id": "prompt_id", "last_node_id": "node_id", "sockets_metadata": "sockets_metadata"}
            if name in fields:
                payload[fields[name]] = value
                return
            if name in ("client_id", "workflow_metadata"):
                payload["extra_data"][name] = value
                return
        object.__setattr__(self, name, value)


def run():
    config = json.loads(sys.stdin.buffer.readline())
    device = config.get("device", "default")
    if device.startswith("cuda:"):
        index = int(device.split(":")[1])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical = visible.split(",")[index] if visible else str(index)
        os.environ["CUDA_VISIBLE_DEVICES"] = physical
        os.environ["HIP_VISIBLE_DEVICES"] = physical
    sys.argv = ["main.py", *config["argv"]]
    connection = Client(tuple(config["address"]), authkey=bytes.fromhex(config["authkey"]))
    send_lock = threading.Lock()

    def send(message):
        with send_lock:
            connection.send(message)

    parent = psutil.Process(os.getppid())

    def watch_parent():
        try:
            while parent.is_running() and parent.status() != psutil.STATUS_ZOMBIE:
                time.sleep(1)
        except psutil.NoSuchProcess:
            pass
        os._exit(0)

    threading.Thread(target=watch_parent, daemon=True).start()

    # Bootstrap the same paths, built-in nodes and custom nodes as the server.
    # These imports must follow device selection and CLI configuration.
    import main
    import execution
    import nodes
    import folder_paths
    import server
    import torch
    import comfy.model_management
    import comfy.model_prefetch
    import comfy.memory_management
    import comfy_aimdo.model_vbar
    from comfy.cli_args import args
    from comfy_api.internal import _ComfyNodeInternal, _NodeOutputInternal, make_locked_method_func
    from comfy_api.latest import _io
    from comfy_execution.graph import DynamicPrompt, ExecutionBlocker
    from comfy_execution.graph_utils import GraphBuilder
    from comfy_execution.progress import ProgressRegistry, scoped_progress_registry, add_progress_handler, WebUIProgressHandler
    from comfy_execution.utils import CurrentNodeContext
    from app.assets.manager import NoAssets
    from latent_preview import set_preview_method

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    worker_server = server.PromptServer(loop, NoAssets(args))
    worker_server.send_sync = lambda event, data, sid=None: send(("event", (event, data, sid)))
    loop.run_until_complete(nodes.init_extra_nodes(
        init_custom_nodes=not args.disable_all_custom_nodes or bool(args.whitelist_custom_nodes),
        init_api_nodes=not args.disable_partner_nodes,
    ))
    worker_server = CallServer(worker_server)
    server.PromptServer.instance = worker_server
    if config["role"] == "node" and args.async_node_concurrency != 1:
        args.disable_comfy_compiler = True
    main.hijack_progress(worker_server)
    comfy.model_management.set_cudnn_benchmark()
    interrupt_lock = threading.Lock()
    active_calls = {}

    def is_cancelled(prompt_id):
        return os.path.exists(os.path.join(config["lock_directory"], prompt_id + ".cancel"))

    def watch_interrupt():
        while True:
            with interrupt_lock:
                for prompt_id, event in active_calls.values():
                    if is_cancelled(prompt_id):
                        event.set()
            time.sleep(0.05)

    threading.Thread(target=watch_interrupt, daemon=True).start()
    send(("ready", None))

    if config["role"] == "prompt":
        from comfy_execution.resident_graph import ResidentGraph, RevisionCache

        graphs = OrderedDict()
        tasks = set()
        channel = BrokerChannel(config)

        async def revise(request_id, payload, graph):
            item = payload["item"]
            extra_data = dict(item[3], **item[5])
            token = call_payload.set({"prompt_id": item[1], "node_id": None, "extra_data": extra_data, "sockets_metadata": payload["sockets_metadata"]})
            interrupted = threading.Event()
            interrupt_token = comfy.model_management.processing_interrupt_event.set(interrupted)
            with interrupt_lock:
                active_calls[request_id] = (item[1], interrupted)
            runtime = Runtime(dict(config, owner=graph.owner), worker_server, channel)
            runtime.prompt_id, runtime.extra_data = item[1], extra_data
            runtime_token = active_runtime.set(runtime)
            scoped_progress_registry.set(ProgressRegistry(item[1], DynamicPrompt(item[2])))
            executor = execution.PromptExecutor(worker_server, cache_type=execution.CacheType.NONE if args.cache_none else execution.CacheType.LRU,
                                                cache_args={"lru": graph.cache_size, "ram": 0, "ram_inactive": 0}, asset_manager=NoAssets(args))
            if not args.cache_none:
                executor.caches.outputs = RevisionCache(graph)
                executor.caches.all = [executor.caches.outputs, executor.caches.objects]
            try:
                await executor.execute_async(item[2], item[1], extra_data, item[4])
                if not executor.success:
                    cancelled = is_cancelled(item[1])
                    await asyncio.wrap_future(channel.submit("cancel", {"prompt_id": item[1]}))
                    for task in runtime.tasks:
                        task.cancel()
                    await asyncio.gather(*runtime.tasks, return_exceptions=True)
                    if not cancelled:
                        graph.outputs.clear()
                result = (executor.history_result, executor.success, executor.status_messages)
                response = ("result", (request_id, result))
            except (Exception, comfy.model_management.InterruptProcessingException):
                graph.outputs.clear()
                response = ("error", (request_id, traceback.format_exc()))
            finally:
                try:
                    async with graph.gate:
                        graph.active -= 1
                        if not graph.active:
                            await asyncio.wrap_future(channel.submit("trim", {
                                "owner": graph.owner, "objects": graph.references(), "session_id": session_key(extra_data),
                                "nodes": [(node_id, node["class_type"]) for node_id, node in graph.prompt.items()],
                            }))
                except Exception:
                    response = ("error", (request_id, traceback.format_exc()))
                send(response)
                with interrupt_lock:
                    active_calls.pop(request_id, None)
                active_runtime.reset(runtime_token)
                comfy.model_management.processing_interrupt_event.reset(interrupt_token)
                call_payload.reset(token)

        async def receive_inputs():
            while True:
                request_id, command, payload = await asyncio.to_thread(connection.recv)
                item = payload["item"]
                key = session_key(item[3]) or (item[3].get("client_id", ""), None)
                if key not in graphs:
                    graphs[key] = ResidentGraph(args.cache_lru or 128)
                graph = graphs[key]
                graphs.move_to_end(key)
                async with graph.gate:
                    graph.prompt = item[2]
                    graph.active += 1
                task = asyncio.create_task(revise(request_id, payload, graph))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                while len(graphs) > 256:
                    oldest = next((key for key, value in graphs.items() if not value.active), None)
                    if oldest is None:
                        break
                    evicted = graphs.pop(oldest)
                    await asyncio.wrap_future(channel.submit("release", {"owner": evicted.owner}))

        loop.run_until_complete(receive_inputs())
        return

    objects = {}
    instances = {}
    next_key = 0
    leases = {}
    model_cache = OrderedDict()
    model_loads = 0
    model_hits = 0
    state_lock = threading.RLock()
    worker_id = config["worker_id"]
    generation = config["generation"]
    loaders = {
        nodes.CheckpointLoaderSimple: {"ckpt_name": "checkpoints"},
        nodes.UNETLoader: {"unet_name": "diffusion_models"},
        nodes.CLIPLoader: {"clip_name": "text_encoders"},
        nodes.DualCLIPLoader: {"clip_name1": "text_encoders", "clip_name2": "text_encoders"},
        nodes.VAELoader: {"vae_name": "vae"},
    }

    def loader_key(class_def, payload):
        if class_def not in loaders or payload["func"] != class_def.FUNCTION:
            return None
        files = []
        for name, category in loaders[class_def].items():
            path = folder_paths.get_full_path(category, payload["inputs"][name])
            if path is None:
                return None
            stat = os.stat(path)
            files.append((path, stat.st_size, stat.st_mtime_ns))
        recipe = json.dumps([payload["class_type"], payload["inputs"]], sort_keys=True)
        return recipe, tuple(files)

    def collect_unused():
        keep = set().union(*leases.values()) if leases else set()
        for key in list(objects):
            if key not in keep:
                del objects[key]

    def encode(value):
        nonlocal next_key
        if value is None or type(value) in (bool, int, float, str, bytes) or isinstance(value, ExecutionBlocker):
            return value
        if isinstance(value, torch.Tensor):
            value = value.detach()
        with state_lock:
            next_key += 1
            objects[next_key] = value
            leases.setdefault(call_payload.get()["owner"], set()).add(next_key)
            return ObjectRef(worker_id, next_key, isinstance(value, torch.Tensor), generation)

    def decode(value):
        if isinstance(value, ObjectRef):
            if value.worker != worker_id or value.generation != generation:
                raise RuntimeError("Stale GPU object reference")
            with state_lock:
                leases.setdefault(call_payload.get()["owner"], set()).add(value.key)
                return objects[value.key]
        if isinstance(value, ObjectData):
            return pickle.loads(value.data)
        return value

    output_lock = FileLock(os.path.join(config["lock_directory"], "outputs.lock"))

    def cleanup():
        if comfy.memory_management.aimdo_enabled:
            comfy.model_prefetch.cleanup_prefetch_queues()
            comfy.model_management.reset_cast_buffers()
            comfy_aimdo.model_vbar.vbars_reset_watermark_limits()

    async def call(payload, obj, inputs):
        nonlocal model_loads, model_hits
        class_type = payload["class_type"]
        node_id = payload["node_id"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        func = payload["func"]
        if isinstance(obj, _ComfyNodeInternal):
            class_def.VALIDATE_CLASS()
            clone = class_def.PREPARE_CLASS_CLONE(payload["v3_data"])
            f = make_locked_method_func(class_def, func, clone)
            inputs = _io.build_nested_inputs(inputs, payload["v3_data"], input_is_list=getattr(obj, "INPUT_IS_LIST", False))
        else:
            f = getattr(obj, func)
        GraphBuilder.set_default_prefix(node_id, payload["index"] or 0, 0)
        set_preview_method(payload["extra_data"].get("preview_method"))
        registry = ProgressRegistry(payload["prompt_id"], DynamicPrompt(payload["prompt"]))
        scoped_progress_registry.set(registry)
        add_progress_handler(WebUIProgressHandler(worker_server))
        cache_key = loader_key(class_def, payload)
        with state_lock:
            if cache_key in model_cache:
                model_hits += 1
                model_cache.move_to_end(cache_key)
                return map_values(tuple(value.clone() if value is not None else None for value in model_cache[cache_key]), encode)
        with CurrentNodeContext(payload["prompt_id"], node_id, payload["index"]):
            # Inference state is process-local, so preserve the executor's scope here.
            with torch.inference_mode():
                result = await f(**inputs) if inspect.iscoroutinefunction(f) else f(**inputs)
        if cache_key is not None:
            with state_lock:
                model_loads += 1
                model_cache[cache_key] = result
                while len(model_cache) > 8:
                    model_cache.popitem(last=False)
            result = tuple(value.clone() if value is not None else None for value in result)
        if isinstance(result, _NodeOutputInternal):
            output = result.result
            if result.block_execution is not None:
                output = tuple(ExecutionBlocker(result.block_execution) for _ in obj.RETURN_TYPES)
            result_dict = {"result": output}
            if result.ui is not None:
                result_dict["ui"] = result.ui if isinstance(result.ui, dict) else result.ui.as_dict()
            if result.expand is not None:
                result_dict["expand"] = result.expand
            result = result_dict
        if isinstance(result, dict) and "ui" in result:
            return {key: value if key == "ui" else map_values(value, encode) for key, value in result.items()}
        return map_values(result, encode)

    # This module is loaded only after the worker's device/bootstrap configuration.
    from comfy_execution.async_inference import NodeAdmission
    admission = NodeAdmission(args.async_node_concurrency, cleanup)

    def stats():
        with state_lock:
            return {"model_loads": model_loads, "model_hits": model_hits,
                    "model_keys": [key[0] for key in model_cache], "objects": len(objects), "owners": len(leases),
                    "active_calls": admission.active, "peak_concurrent_calls": admission.peak,
                    "concurrency_limit": admission.limit,
                    "resident_preparations": admission.preparations}

    def execute_call(request_id, payload, admitted):
        payload = dict(payload, call_id=request_id)
        token = call_payload.set(payload)
        interrupted = threading.Event()
        if is_cancelled(payload["prompt_id"]):
            interrupted.set()
        interrupt_token = comfy.model_management.processing_interrupt_event.set(interrupted)
        with interrupt_lock:
            active_calls[request_id] = (payload["prompt_id"], interrupted)
        key = (payload["owner"], payload["session_id"], payload["node_id"], payload["class_type"])
        obj = None
        stream = None
        try:
            class_def = nodes.NODE_CLASS_MAPPINGS[payload["class_type"]]
            inputs = map_values(payload["inputs"], decode)
            is_output = getattr(class_def, "OUTPUT_NODE", False)
            retry_safe = admission.sampler_memory(class_def, payload["func"], inputs) is not None or getattr(class_def, "ASYNC_RETRY_SAFE", False)
            attempts = 3 if retry_safe and not is_output else 1
            device = comfy.model_management.get_torch_device()
            if device.type == "cuda":
                stream = torch.cuda.Stream(device=device)
            for attempt in range(attempts):
                try:
                    with admission.execution(class_def, payload["func"], inputs, serial=attempt > 0) as call_inputs:
                        admitted.set()
                        if obj is None:
                            with state_lock:
                                pool = instances.setdefault(key, [])
                                obj = pool.pop() if pool else class_def()
                        details = {"prompt_id": payload["prompt_id"], "node_id": payload["node_id"],
                                   "call_id": request_id, "list_index": payload["index"], "stream": stream.cuda_stream if stream else None,
                                   "active_calls": admission.active}
                        try:
                            send(("event", ("async_node_start", details, worker_server.client_id)))
                            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                                with output_lock if is_output else nullcontext():
                                    nodes.before_node_execution()
                                    try:
                                        result = asyncio.run(call(payload, obj, call_inputs))
                                    finally:
                                        if stream is not None:
                                            stream.synchronize()
                        finally:
                            send(("event", ("async_node_end", details, worker_server.client_id)))
                    break
                except Exception as exc:
                    if comfy.model_management.is_oom(exc):
                        admission.recover()
                    if not comfy.model_management.is_oom(exc) or attempt + 1 == attempts or interrupted.is_set():
                        raise
                    time.sleep(0.1 * (2 ** attempt))
            send(("result", (request_id, {"value": result, "stats": stats()})))
        except (Exception, comfy.model_management.InterruptProcessingException):
            send(("error", (request_id, traceback.format_exc())))
        finally:
            admitted.set()
            if obj is not None:
                with state_lock:
                    instances[key].append(obj)
            with interrupt_lock:
                active_calls.pop(request_id, None)
            comfy.model_management.processing_interrupt_event.reset(interrupt_token)
            call_payload.reset(token)

    pending = []
    while True:
        request_id, command, payload = connection.recv()
        if command == "call":
            pending = [thread for thread in pending if thread.is_alive()]
            admitted = threading.Event()
            thread = threading.Thread(target=execute_call, args=(request_id, payload, admitted))
            pending.append(thread)
            thread.start()
            # Keep only one new call waiting for memory or an exclusive node.
            admitted.wait()
            continue
        # Objects are exported or retired only after earlier calls finish.
        for thread in pending:
            thread.join()
        pending.clear()
        try:
            owner = payload.get("owner")
            if command == "export":
                result = ObjectData(pickle.dumps(objects[payload["ref"].key].detach().cpu(), protocol=pickle.HIGHEST_PROTOCOL))
            else:
                if command == "release":
                    leases.pop(owner, None)
                else:
                    leases[owner] = {ref.key for ref in payload["objects"] if ref.worker == worker_id and ref.generation == generation}
                for key in list(instances):
                    if key[0] == owner and (command == "release" or key[1] != payload["session_id"] or key[2:] not in payload["nodes"]):
                        del instances[key]
                collect_unused()
                result = None
            send(("result", (request_id, {"value": result, "stats": stats()})))
        except Exception:
            send(("error", (request_id, traceback.format_exc())))


if __name__ == "__main__":
    run()
