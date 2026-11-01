"""Shared completed outputs with a private cache view for each input revision."""

import asyncio
import secrets
from collections import OrderedDict

from comfy_execution.async_execution import ObjectRef, map_values
from comfy_execution.caching import CacheKeySetInputSignature, LRUCache


class RevisionCache(LRUCache):
    def __init__(self, graph):
        super().__init__(CacheKeySetInputSignature, max_size=graph.cache_size, enable_providers=True)
        self.graph = graph

    def get_local(self, node_id):
        value = super().get_local(node_id)
        if value is None and self.initialized:
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
        while len(self.graph.outputs) > self.graph.cache_size:
            self.graph.outputs.popitem(last=False)

    async def _set_immediate(self, node_id, value):
        await super()._set_immediate(node_id, value)
        self.publish(node_id, value)

    def set_local(self, node_id, value):
        super().set_local(node_id, value)
        self.publish(node_id, value)


class ResidentGraph:
    def __init__(self, cache_size):
        self.owner = secrets.token_hex(16)
        self.cache_size = cache_size
        self.outputs = OrderedDict()
        self.active = 0
        self.gate = asyncio.Lock()
        self.prompt = {}

    def references(self):
        refs = set()

        def collect(value):
            if isinstance(value, ObjectRef):
                refs.add(value)
            return value

        for entry in self.outputs.values():
            map_values(entry.outputs, collect)
        return list(refs)
