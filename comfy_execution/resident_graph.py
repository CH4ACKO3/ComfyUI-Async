"""Shared completed outputs with a private cache view for each input revision."""

import asyncio
import secrets
from collections import OrderedDict

from comfy_execution.async_execution import ObjectRef, map_values
from comfy_execution.caching import CacheKeySetInputSignature, LRUCache


class RevisionCache(LRUCache):
    def __init__(self, graph, revision):
        super().__init__(CacheKeySetInputSignature, max_size=graph.cache_size, enable_providers=True)
        self.graph = graph
        self.revision = revision
        graph.views.add(self)

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        await super().set_prompt(dynprompt, node_ids, is_changed_cache)
        if self.revision == self.graph.revision:
            self.graph.current_keys = set(self.cache_key_set.get_used_keys())
            self.graph.prune()

    def get_local(self, node_id):
        value = super().get_local(node_id)
        if value is None and self.initialized:
            if "_loop_end" in self.dynprompt.get_node(node_id):
                return None
            key = self.cache_key_set.get_data_key(node_id)
            value = self.graph.outputs.get(key)
            if value is not None:
                self.graph.outputs.move_to_end(key)
                self.cache[key] = value
        return value

    async def _get_immediate(self, node_id):
        self.get_local(node_id)
        return await super()._get_immediate(node_id)

    def publish(self, node_id, value):
        key = self.cache_key_set.get_data_key(node_id)
        self.graph.outputs[key] = value
        self.graph.outputs.move_to_end(key)
        self.graph.prune()

    async def _set_immediate(self, node_id, value):
        await super()._set_immediate(node_id, value)
        self.publish(node_id, value)

    def set_local(self, node_id, value):
        super().set_local(node_id, value)
        self.publish(node_id, value)


class NodeCall:
    def __init__(self):
        self.id = secrets.token_hex(16)
        self.ready = asyncio.Event()
        self.users = set()
        self.task = None


class ResidentGraph:
    def __init__(self, cache_size):
        self.owner = secrets.token_hex(16)
        self.cache_size = cache_size
        self.outputs = OrderedDict()
        self.calls = {}
        self.leased = set()
        self.views = set()
        self.current_keys = set()
        self.revision = 0
        self.active = 0
        self.gate = asyncio.Lock()
        self.prompt = {}

    async def dispatch(self, call, channel, payload):
        await call.ready.wait()
        result = await asyncio.wrap_future(channel.submit("call", dict(payload, prompt_id=call.id)))
        self.collect_refs(result, self.leased)
        return result

    async def call(self, runtime, node_id, index, func, payload):
        signature = runtime.node_keys[node_id] if func != "check_lazy_status" else None
        key = (signature, node_id, index, func) if signature is not None else (None, secrets.token_hex(16))
        call = self.calls.get(key)
        if call is None:
            call = self.calls[key] = NodeCall()
            call.task = asyncio.create_task(self.dispatch(call, runtime.channel, payload))
        call.users.add(runtime.prompt_id)
        subscription = {"call_id": call.id, "prompt_id": runtime.prompt_id}
        result = asyncio.shield(call.task)

        async def wait_cancel():
            while not runtime.interrupted.is_set():
                await asyncio.sleep(0.05)

        cancelled = asyncio.create_task(wait_cancel())
        try:
            await asyncio.wrap_future(runtime.channel.submit("subscribe", subscription))
            call.ready.set()
            done, _ = await asyncio.wait((result, cancelled), return_when=asyncio.FIRST_COMPLETED)
            if cancelled in done:
                raise asyncio.CancelledError()
            return result.result()
        finally:
            cancelled.cancel()
            result.cancel()
            call.users.discard(runtime.prompt_id)
            await asyncio.wrap_future(runtime.channel.submit("unsubscribe", subscription))
            if not call.users:
                if not call.task.done():
                    if self.calls.get(key) is call:
                        del self.calls[key]
                    await asyncio.wrap_future(runtime.channel.submit("cancel", {"prompt_id": call.id}))
                    call.task.cancel()
                    await asyncio.gather(call.task, return_exceptions=True)
                if (call.task.cancelled() or call.task.exception() is not None) and self.calls.get(key) is call:
                    del self.calls[key]
            self.prune()

    def invalidate_device(self, device):
        def stale(value):
            refs = set()
            self.collect_refs(value, refs)
            return any(ref.worker == device for ref in refs)

        for cache in [self.outputs, *(view.cache for view in self.views)]:
            for key, entry in list(cache.items()):
                if stale(entry.outputs):
                    del cache[key]
        for key, call in list(self.calls.items()):
            if call.task.done() and not call.task.cancelled() and call.task.exception() is None and stale(call.task.result()):
                del self.calls[key]
        self.leased = {ref for ref in self.leased if ref.worker != device}

    def prune(self):
        live = self.current_keys.copy()
        for view in self.views:
            if view.initialized:
                live.update(view.cache_key_set.get_used_keys())
        old = [key for key in self.outputs if key not in live]
        for key in old[:max(0, len(old) - self.cache_size)]:
            del self.outputs[key]
        for key, call in list(self.calls.items()):
            if not call.users and call.task.done() and key[0] not in live and key[0] not in self.outputs:
                del self.calls[key]

    @staticmethod
    def collect_refs(value, refs):
        def collect(value):
            if isinstance(value, ObjectRef):
                refs.add(value)
            return value
        map_values(value, collect)

    def references(self):
        refs = set()
        for entry in self.outputs.values():
            self.collect_refs(entry.outputs, refs)
        for view in self.views:
            for entry in view.cache.values():
                self.collect_refs(entry.outputs, refs)
        for call in self.calls.values():
            if call.task.done() and not call.task.cancelled() and call.task.exception() is None:
                self.collect_refs(call.task.result(), refs)
        return refs

    async def retire(self, channel):
        refs = self.references()
        retired = self.leased - refs
        if retired:
            await asyncio.wrap_future(channel.submit("retire", {"owner": self.owner, "objects": list(retired)}))
            self.leased.difference_update(retired)
