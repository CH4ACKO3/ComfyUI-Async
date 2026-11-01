"""Admission and resident-model leases for concurrent calls in a device worker."""

import copy
import threading
from contextlib import contextmanager

import torch

import nodes
import comfy.model_management as mm
import comfy.model_patcher
import comfy.sampler_helpers
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel


class NodeAdmission:
    def __init__(self, limit, cleanup):
        self.limit = limit
        self.cleanup = cleanup
        self.condition = threading.Condition()
        self.active = 0
        self.exclusive = False
        self.draining = False
        self.writers = 0
        self.reserved = 0
        self.peak = 0
        self.prepared = None
        self.preparations = 0

    def sampler_memory(self, class_def, func, inputs):
        if class_def not in (nodes.KSampler, nodes.KSamplerAdvanced) or func != class_def.FUNCTION:
            return None
        model = inputs['model']
        if (mm.vram_state not in (mm.VRAMState.NORMAL_VRAM, mm.VRAMState.HIGH_VRAM)
                or type(model) not in (comfy.model_patcher.ModelPatcher, comfy.model_patcher.ModelPatcherDynamic)
                or model.load_device.type != 'cuda' or type(model.model.diffusion_model) is not UNetModel
                or model.patches or model.object_patches or model.weight_wrapper_patches or model.hook_patches
                or model.attachments or model.additional_models or model.injections
                or model.model_options != {'transformer_options': {}}
                or any(items for group in model.callbacks.values() for items in group.values())
                or any(items for group in model.wrappers.values() for items in group.values())
                or inputs['sampler_name'] != 'euler'):
            return None
        conds = {}
        for name in ('positive', 'negative'):
            if any(set(metadata) - {'pooled_output'} for _, metadata in inputs[name]):
                return None
            conds[name] = comfy.sampler_helpers.convert_cond(inputs[name])
        shape = inputs['latent_image']['samples'].shape
        required, _ = comfy.sampler_helpers.estimate_memory(model, shape, conds)
        return max(required, 512 * 1024 * 1024)

    def prepare(self, model, memory):
        if model.model_size() + memory + mm.extra_reserved_memory() > mm.get_total_memory(model.load_device):
            return None
        key = (model.model, model.patches_uuid)
        if self.prepared is None or self.prepared[:2] != key:
            resident = model.get_non_dynamic_delegate() if model.is_dynamic() else model
            self.prepared = (*key, resident)
            self.preparations += 1
        resident = self.prepared[2]
        mm.load_models_gpu([resident], memory_required=memory, force_full_load=True)
        torch.cuda.current_stream(model.load_device).synchronize()
        if resident.model.model_lowvram or resident.loaded_size() < resident.model_size():
            return None
        return resident

    @contextmanager
    def execution(self, class_def, func, inputs, serial=False):
        sampler_memory = None if serial or self.limit == 1 else self.sampler_memory(class_def, func, inputs)
        safe_parallel = not serial and self.limit != 1 and (
            class_def in (nodes.EmptyImage, nodes.EmptyLatentImage, nodes.ImageScale, nodes.ImageScaleBy, nodes.ImageInvert)
            or getattr(class_def, 'RELATIVE_PYTHON_MODULE', None) == 'comfy_extras.nodes_primitive'
            or getattr(class_def, 'ASYNC_CONCURRENT_SAFE', False))
        device = mm.get_torch_device()
        resident = None
        with self.condition:
            while True:
                mm.throw_exception_if_processing_interrupted()
                if (not self.exclusive and not self.draining and not self.writers
                        and (self.limit is None or self.active < self.limit)):
                    parallel = safe_parallel
                    memory = 512 * 1024 * 1024 if parallel else 0
                    resident = None
                    if sampler_memory is not None:
                        model = inputs['model']
                        if self.prepared is not None and self.prepared[:2] == (model.model, model.patches_uuid):
                            candidate = self.prepared[2]
                            if candidate.loaded_size() >= candidate.model_size() and not candidate.model.model_lowvram:
                                resident = candidate
                        if self.active == 0:
                            resident = self.prepare(model, sampler_memory)
                        if resident is not None:
                            parallel = True
                            memory = sampler_memory
                    if parallel:
                        headroom = mm.extra_reserved_memory() if device.type != 'cpu' else 0
                        if mm.get_free_memory(device) < self.reserved + memory + headroom:
                            if self.active:
                                self.condition.wait(0.05)
                                continue
                            parallel, memory, resident = False, 0, None
                    if parallel or self.active == 0:
                        self.exclusive = not parallel
                        self.active += 1
                        self.reserved += memory
                        self.peak = max(self.peak, self.active)
                        break
                self.condition.wait(0.05)
        token = None
        try:
            if parallel:
                token = mm.resident_model_scope.set(())
            if resident is not None:
                # Sampler state lives on the outer model and patcher; the resident
                # UNet and its immutable parameters are shared by these calls.
                outer = copy.copy(resident.model)
                patcher = resident.clone(model_override=(outer, ({}, {}, {}, set())))
                inputs = dict(inputs, model=patcher)
                mm.resident_model_scope.set((patcher,))
            yield inputs
        finally:
            if token is not None:
                mm.resident_model_scope.reset(token)
            with self.condition:
                self.active -= 1
                self.reserved -= memory
                if resident is not None:
                    self.draining = True
                if self.active == 0:
                    self.exclusive = False
                    self.draining = False
                    self.cleanup()
                self.condition.notify_all()

    def recover(self):
        with self.condition:
            self.writers += 1
            try:
                while self.active:
                    self.condition.wait(0.05)
                mm.unload_all_models()
                mm.soft_empty_cache()
            finally:
                self.writers -= 1
                self.condition.notify_all()
