# Async GPU groups

This fork uses one resident graph coordinator with no fixed limit on concurrent input revisions. Each revision keeps the upstream dependency, lazy-input and dynamic-subgraph scheduler, and dispatches ready nodes to shared device queues. Use `--disable-async-execution` to use the upstream executor.

Right-click a native Group and open **Execution GPUs**. All GPUs are enabled by default; click checkboxes to toggle them without closing the menu. The group header displays the allowed GPUs without changing its saved title. Turning all GPUs off restores native device selection, just like nodes outside groups; it does not force CPU execution or inherit GPU lists from overlapping groups. Selecting several GPUs creates a scheduling pool, not tensor parallelism. Settings are saved in the group's `flags.gpu_devices` and exported to each node's `_meta.gpu_devices` in API prompts. A node belongs to the smallest group containing its center (including the boundary); equal-sized groups use the lowest group ID, independent of drawing order. GPU lists are not combined across overlapping groups. Nodes outside all groups retain native device selection and follow the device of input model objects.

Put model loaders and their consumers in the same GPU group. Models and arbitrary custom objects stay in their owning process; separate loaders are needed for separate GPUs. Tensors can cross groups through CPU memory. Native loop controllers remain in the graph coordinator. Node interfaces are unchanged; custom nodes relying on process-global state, custom HTTP routes, or live graph mutation may need the `--disable-async-execution` compatibility path. This version targets local GPUs; CUDA is verified, while ROCm and other backends are unverified. Remote workers are not implemented.

Each GPU has one persistent worker. Node concurrency follows available memory by default, with no fixed call-count limit; `--async-node-concurrency N` optionally caps it (`1` runs serially). Samplers use native activation-memory estimates, and other parallel calls reserve at least 512 MiB. Waiting inputs do not each occupy a compute thread in the device worker. Calls have separate node instances, CUDA streams, progress and cancellation state, including multiple list inputs to the same node. Native checkpoint, UNet, CLIP and VAE loaders share cached base weights through native model clones (up to eight loader configurations per GPU); custom loaders do not participate in this cache.

Concurrent sampling currently supports plain UNet-based SD/SDXL `KSampler` and `KSamplerAdvanced` with Euler, ordinary conditioning and no model patches, hooks or additional models. A native full-load delegate keeps one shared set of resident GPU weights; each invocation gets its own sampler/model wrapper and an activation-memory reservation. Loading, unloading and incompatible calls wait for active readers. Low/no-VRAM mode, models that cannot fit the reservation, and unsupported sampling configurations retain native serial execution. The Comfy compiler is disabled in concurrent device workers because its shared state is not reentrant; concurrency `1` retains the native compiler setting.

Basic image operations can overlap, and custom nodes can declare `ASYNC_CONCURRENT_SAFE = True` when isolated instances can run without mutating shared inputs, process globals or managed-model residency. Unmarked custom nodes remain exclusive. Output nodes share a lock to prevent filename collisions. Eligible plain samplers and non-output nodes declaring `ASYNC_RETRY_SAFE = True` receive up to three attempts on OOM, with backoff and serial retries after active calls drain. Other failed nodes are not replayed. A model that cannot fit still fails normally.

Completed outputs use a per-session LRU cache shared across revisions (128 entries by default, configurable with `--cache-lru`; `--cache-none` disables it). Each revision has its own dependency and cache-key state. This replaces the upstream RAM-pressure cache because coordinators cannot measure remote object storage. Changing a GPU assignment invalidates affected cached results. Interrupting an input signals only that request while retaining shared outputs, workers and weights. Custom nodes must cooperate with native interruption; a blocking custom call can delay cancellation. The native free-memory action clears workers and caches once active inputs finish. `GET /async/status` reports worker PIDs, pending calls, peak concurrency and model-cache counters. Worker progress/preview events include `call_id` and `list_index`; `async_node_start`/`async_node_end` identify individual invocations.

## Continuous inputs

Native Run and auto-queue modes continue to work. API clients can also include this field in `POST /prompt` alongside the usual prompt and client ID:

```json
{"extra_data": {"execution_session": {"id": "canvas-1", "revision": 1, "policy": "latest"}}}
```

Use a stable client/session ID and an increasing revision for each input. `latest` replaces inputs still waiting in the submission queue; already admitted inputs continue. `fifo` admits every input in queue order and returns HTTP 429 at 64 pending submissions per session. Both policies allow concurrent revisions and out-of-order completion. Revision tracking and graph caches retain up to 256 idle/active sessions, evicting only idle sessions.

After submitting the full graph once, refresh existing nodes through the same `POST /prompt` endpoint without a `prompt` field:

```json
{"client_id": "my-client", "extra_data": {"execution_session": {"id": "canvas-1", "revision": 2, "policy": "fifo"}}, "refresh": {"8": {"value": 123}}}
```

The client/session must match the initial submission. Each refresh merges the supplied inputs and invalidates that node and its descendants; an empty input mapping forces recalculation with the same values. Unchanged completed branches can reuse cached outputs. Existing node implementations require no refresh-specific changes.

Progress, execution and metadata-bearing preview events include `execution_session`. Replaced inputs receive an `execution_superseded` event and a matching unsuccessful history entry. Clients should display only results matching their desired revision. Image inputs must use distinct filenames per revision so later uploads cannot overwrite queued snapshots. This API does not yet add a painting node or replace the frontend's existing auto-queue triggers.
