import { app } from "/scripts/app.js";
import { api, PromptExecutionError } from "/scripts/api.js";

const graphs = new Map();
let timer;
let pendingEdit;
let configuring = false;
let queuedWidgetChange = false;

function canonical(value) {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === "object") {
        return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
    }
    return value;
}

function signature(node) {
    return JSON.stringify(canonical([node.class_type, node.inputs, node._meta?.gpu_devices ?? []]));
}

function differences(previous, current) {
    const changes = {};
    for (const [id, node] of Object.entries(current)) {
        if (!previous[id] || signature(previous[id]) !== signature(node)) changes[id] = node;
    }
    for (const id of Object.keys(previous)) if (!current[id]) changes[id] = null;
    return changes;
}

function graphState(snapshot) {
    const key = `${api.clientId ?? ""}:${snapshot.workflow.id}`;
    let state = graphs.get(key);
    if (!state) {
        state = { id: LiteGraph.uuidv4(), revision: 0, sent: null, observed: snapshot.output, pending: Promise.resolve() };
        graphs.set(key, state);
    }
    return state;
}

async function request(path, method, body) {
    const response = await api.fetchApi(path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) throw new PromptExecutionError(result, response.status);
    return result;
}

function submit(snapshot, refresh, options = {}, number = 0) {
    const state = graphState(snapshot);
    const client = api.clientId ?? "";
    const auth = { auth_token_comfy_org: api.authToken, api_key_comfy_org: api.apiKey };
    const operation = state.pending.then(async () => {
        if (!state.sent) {
            await request(`/graphs/${state.id}`, "PUT", { client_id: client, nodes: snapshot.output });
            state.sent = snapshot.output;
        }
        const changes = differences(state.sent, snapshot.output);
        if (!Object.keys(changes).length && !refresh.length) return null;
        const result = await request(`/graphs/${state.id}/events`, "POST", {
            client_id: client,
            event_id: LiteGraph.uuidv4(),
            base_revision: state.revision,
            changes,
            refresh,
            ...(number === -1 ? { number: -Date.now() } : number ? { number } : {}),
            extra_data: {
                ...auth,
                comfy_usage_source: "comfyui-frontend",
                extra_pnginfo: { workflow: snapshot.workflow },
                preview_method: options.previewMethod,
            },
        });
        state.sent = snapshot.output;
        state.revision = result.revision;
        return result;
    });
    state.pending = operation.catch(() => {});
    return operation;
}

async function observe(baseline = false) {
    if (configuring) return;
    const snapshot = structuredClone(await app.graphToPrompt());
    const state = graphState(snapshot);
    const changes = differences(state.observed, snapshot.output);
    state.observed = snapshot.output;
    if (baseline || !Object.keys(changes).length) return;
    const refresh = pendingEdit?.state === state ? pendingEdit.refresh : new Set();
    for (const id of Object.keys(changes)) refresh.add(id);
    pendingEdit = { state, snapshot, refresh };
    clearTimeout(timer);
    timer = setTimeout(async () => {
        const edit = pendingEdit;
        pendingEdit = null;
        try {
            await submit(edit.snapshot, [...edit.refresh].filter((id) => edit.snapshot.output[id]));
        } catch (error) {
            app.extensionManager.toast.add({ severity: "error", summary: "Node update failed", detail: error.response?.error?.message ?? error.message, life: 5000 });
        }
    }, 150);
}

app.registerExtension({
    name: "Comfy.Async.GraphEvents",
    beforeConfigureGraph() {
        configuring = true;
        clearTimeout(timer);
        pendingEdit = null;
    },
    afterConfigureGraph() {
        configuring = false;
        void observe(true);
    },
    async setup() {
        api.queuePrompt = async function (number, data, options = {}) {
            clearTimeout(timer);
            pendingEdit = null;
            const snapshot = structuredClone(data);
            graphState(snapshot).observed = snapshot.output;
            const selected = options.partialExecutionTargets;
            const refresh = selected?.length ? selected.map(String) : Object.keys(snapshot.output).filter((id) =>
                !Object.values(snapshot.output[id].inputs).some((value) => Array.isArray(value) && value.length === 2 && typeof value[0] === "string" && Number.isInteger(value[1])));
            if (!refresh.length) refresh.push(...Object.keys(snapshot.output));
            if (!refresh.length) throw new PromptExecutionError({ error: { message: "Graph has no nodes" }, node_errors: {} }, 400);
            return await submit(snapshot, refresh, options, number);
        };

        // Graph edits already submit events, including when the old auto-queue mode is selected.
        const dispatch = api.dispatchCustomEvent;
        api.dispatchCustomEvent = function (type, ...args) {
            if (type === "autoQueueGraphChanged") return;
            const previous = queuedWidgetChange;
            if (type === "promptQueued") queuedWidgetChange = true;
            try {
                return dispatch.call(this, type, ...args);
            } finally {
                queuedWidgetChange = previous;
            }
        };
        api.addEventListener("graphChanged", () => {
            void observe(queuedWidgetChange);
        });
        api.addEventListener("reconnected", () => {
            graphs.clear();
            void observe(true);
        });
        await observe(true);
    },
});
