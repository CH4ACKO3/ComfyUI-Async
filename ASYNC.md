# Async GPU groups

This fork runs up to two workflows concurrently by default. Each workflow keeps the upstream dependency, lazy-input and dynamic-subgraph scheduler, and dispatches ready nodes to isolated worker processes. Use `--async-workers N` to change the workflow limit, or `--async-workers 0` to use the upstream executor.

Right-click a native Group and open **Execution GPUs**. All GPUs are enabled by default; click checkboxes to toggle them without closing the menu. The group header displays the allowed GPUs without changing its saved title. Turning all GPUs off restores native device selection, just like nodes outside groups; it does not force CPU execution or inherit GPU lists from overlapping groups. Selecting several GPUs creates a scheduling pool, not tensor parallelism. Settings are saved in the group's `flags.gpu_devices` and exported to each node's `_meta.gpu_devices` in API prompts. A node belongs to the smallest group containing its center (including the boundary); equal-sized groups use the lowest group ID, independent of drawing order. GPU lists are not combined across overlapping groups. Nodes outside all groups retain native device selection and follow the device of input model objects.

Put model loaders and their consumers in the same GPU group. Models and arbitrary custom objects stay in their owning process; separate loaders are needed for separate GPUs. Tensors can cross groups through CPU memory. Native loop controllers remain in the graph coordinator. Custom nodes using process-global state, custom HTTP routes, or live graph mutation may need adaptation; `--async-workers 0` provides the compatibility path. This version targets local GPUs; CUDA is verified, while ROCm and other backends are unverified. Remote workers are not implemented.

Different GPUs execute simultaneously through one persistent worker per device. All workflows share that worker; calls on the same GPU are serialized. Native checkpoint, UNet, CLIP and VAE loaders share cached base weights through native model clones (up to eight loader configurations per GPU). Patched model wrappers remain separate, and native model management controls VRAM residency and offloading. Custom loaders do not participate in this weight cache. Output nodes share a lock to prevent filename collisions. Failed nodes are not replayed unless their class declares `ASYNC_RETRY_SAFE = True`; non-output nodes with that declaration get up to three attempts on OOM, with bounded backoff. A model that cannot fit still fails normally.

Intermediate outputs use a per-coordinator LRU cache (128 entries by default, configurable with `--cache-lru`; `--cache-none` disables it). This replaces the upstream RAM-pressure cache because coordinators cannot measure remote object storage. Changing a GPU assignment invalidates affected cached results. Interrupting a workflow signals only that request and releases its cached outputs while retaining shared workers and weights. Custom nodes must cooperate with native interruption; a blocking custom call can delay cancellation. The native free-memory action clears workers and caches once active workflows finish. `GET /async/status` reports worker PIDs, pending calls and model-cache counters.

## Continuous inputs

Native Run and auto-queue modes continue to work. API clients can also include this field in `POST /prompt` alongside the usual prompt and client ID:

```json
{"extra_data": {"execution_session": {"id": "canvas-1", "revision": 1, "policy": "latest"}}}
```

Use a stable client/session ID and an increasing revision for each input snapshot. `latest` keeps the running input and replaces pending inputs with the newest one; `fifo` executes every input and returns HTTP 429 at 64 pending inputs per session. One input per session runs at a time, with coordinator affinity to reuse unchanged graph results. Different sessions can run concurrently up to `--async-workers`. Revision tracking retains the 256 most recent sessions, evicting only idle sessions; switching sessions on a coordinator clears its graph cache but retains device model weights.

Progress, execution and metadata-bearing preview events include `execution_session`. Replaced inputs receive an `execution_superseded` event and a matching unsuccessful history entry. Clients should display only results matching their desired revision. Image inputs must use distinct filenames per revision so later uploads cannot overwrite queued snapshots. This API does not yet add a painting node or replace the frontend's existing auto-queue triggers.
