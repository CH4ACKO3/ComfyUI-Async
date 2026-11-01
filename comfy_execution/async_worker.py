"""Private, authenticated local worker. Started by async_execution.Worker."""

import asyncio
import inspect
import json
import os
import pickle
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from multiprocessing.connection import Client

import psutil
from filelock import FileLock

from comfy_execution.async_execution import ObjectData, ObjectRef, Runtime, active_runtime, map_values


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
    from comfy_execution.progress import reset_progress_state, add_progress_handler, WebUIProgressHandler
    from comfy_execution.utils import CurrentNodeContext
    from app.assets.manager import NoAssets

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    worker_server = server.PromptServer(loop, NoAssets(args))
    worker_server.send_sync = lambda event, data, sid=None: send(("event", (event, data, sid)))
    loop.run_until_complete(nodes.init_extra_nodes(
        init_custom_nodes=not args.disable_all_custom_nodes or bool(args.whitelist_custom_nodes),
        init_api_nodes=not args.disable_partner_nodes,
    ))
    main.hijack_progress(worker_server)
    comfy.model_management.set_cudnn_benchmark()
    send(("ready", None))

    if config["role"] == "prompt":
        runtime = Runtime(config, worker_server)
        # Remote objects own their storage; the coordinator cannot measure their RAM.
        cache_type = execution.CacheType.NONE if args.cache_none else execution.CacheType.LRU
        executor = execution.PromptExecutor(worker_server, cache_type=cache_type, cache_args={"lru": args.cache_lru or 128, "ram": 0, "ram_inactive": 0}, asset_manager=NoAssets(args))
        try:
            while True:
                command, payload = connection.recv()
                item = payload["item"]
                extra_data = dict(item[3], **item[5])
                runtime.prompt_id = item[1]
                runtime.extra_data = extra_data
                worker_server.last_prompt_id = item[1]
                worker_server.workflow_metadata = extra_data.get("workflow_metadata", {})
                worker_server.sockets_metadata = payload["sockets_metadata"]
                token = active_runtime.set(runtime)
                try:
                    loop.run_until_complete(executor.execute_async(item[2], item[1], extra_data, item[4]))
                    result = (executor.history_result, executor.success, executor.status_messages)
                    if not executor.success:
                        # Stop sibling nodes before reporting a failed workflow.
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        runtime.close()
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        executor.reset()
                        runtime = Runtime(config, worker_server)
                    else:
                        runtime.trim()
                    send(("result", result))
                finally:
                    active_runtime.reset(token)
        finally:
            runtime.close()
        return

    objects = {}
    instances = {}
    next_key = 0

    def encode(value):
        nonlocal next_key
        if value is None or type(value) in (bool, int, float, str, bytes) or isinstance(value, ExecutionBlocker):
            return value
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        next_key += 1
        objects[next_key] = value
        return ObjectRef(device, next_key, isinstance(value, torch.Tensor), type(value).__name__)

    def decode(value):
        if isinstance(value, ObjectRef):
            return objects[value.key]
        if isinstance(value, ObjectData):
            return pickle.loads(value.data)
        return value

    actual_device = config["default_device"] if device == "default" else device
    gpu_lock = FileLock(os.path.join(config["lock_directory"], actual_device.replace(":", "-") + ".lock"))
    output_lock = FileLock(os.path.join(config["lock_directory"], "outputs.lock"))

    def cleanup():
        if comfy.memory_management.aimdo_enabled:
            comfy.model_prefetch.cleanup_prefetch_queues()
            comfy.model_management.reset_cast_buffers()
            comfy_aimdo.model_vbar.vbars_reset_watermark_limits()
        # Other workflow processes must be able to use this GPU after admission.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

    async def call(payload):
        class_type = payload["class_type"]
        node_id = payload["node_id"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        key = (node_id, class_type)
        if key not in instances:
            instances[key] = class_def()
        obj = instances[key]
        inputs = map_values(payload["inputs"], decode)
        func = payload["func"]
        if isinstance(obj, _ComfyNodeInternal):
            class_def.VALIDATE_CLASS()
            clone = class_def.PREPARE_CLASS_CLONE(payload["v3_data"])
            f = make_locked_method_func(class_def, func, clone)
            inputs = _io.build_nested_inputs(inputs, payload["v3_data"], input_is_list=getattr(obj, "INPUT_IS_LIST", False))
        else:
            f = getattr(obj, func)
        GraphBuilder.set_default_prefix(node_id, payload["index"] or 0, 0)
        worker_server.client_id = payload["extra_data"].get("client_id")
        worker_server.last_prompt_id = payload["prompt_id"]
        worker_server.last_node_id = node_id
        worker_server.workflow_metadata = payload["extra_data"].get("workflow_metadata", {})
        worker_server.sockets_metadata = payload["sockets_metadata"]
        reset_progress_state(payload["prompt_id"], DynamicPrompt(payload["prompt"]))
        add_progress_handler(WebUIProgressHandler(worker_server))
        nodes.interrupt_processing(False)
        with CurrentNodeContext(payload["prompt_id"], node_id, payload["index"]):
            # Inference state is process-local, so preserve the executor's scope here.
            with torch.inference_mode():
                result = await f(**inputs) if inspect.iscoroutinefunction(f) else f(**inputs)
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

    while True:
        command, payload = connection.recv()
        try:
            if command == "export":
                result = ObjectData(pickle.dumps(objects[payload], protocol=pickle.HIGHEST_PROTOCOL))
            elif command == "trim":
                keep = set(payload["objects"])
                for key in list(objects):
                    if key not in keep:
                        del objects[key]
                keep_nodes = set(payload["nodes"])
                for key in list(instances):
                    if key not in keep_nodes:
                        del instances[key]
                result = None
            else:
                class_def = nodes.NODE_CLASS_MAPPINGS[payload["class_type"]]
                is_output = getattr(class_def, "OUTPUT_NODE", False)
                attempts = 3 if getattr(class_def, "ASYNC_RETRY_SAFE", False) and not is_output else 1
                for attempt in range(attempts):
                    try:
                        with gpu_lock, output_lock if is_output else nullcontext():
                            try:
                                result = loop.run_until_complete(call(payload))
                            finally:
                                cleanup()
                        break
                    except Exception as exc:
                        if not comfy.model_management.is_oom(exc) or attempt + 1 == attempts:
                            raise
                        time.sleep(0.1 * (2 ** attempt))
            send(("result", result))
        except Exception:
            send(("error", traceback.format_exc()))


if __name__ == "__main__":
    run()
