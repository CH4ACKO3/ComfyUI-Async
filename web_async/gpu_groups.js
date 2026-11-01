import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "Comfy.Async.GPUGroups",
    async setup() {
        const response = await api.fetchApi("/async/devices");
        if (!response.ok) return;
        const { enabled, devices } = await response.json();
        if (!enabled) return;

        const groupPrototype = LiteGraph.LGraphGroup.prototype;
        const getMenuOptions = groupPrototype.getMenuOptions;
        groupPrototype.getMenuOptions = function (...args) {
            const selected = this.flags.gpu_devices ?? [];
            const change = (next) => {
                this.graph?.beforeChange();
                if (next.length) this.flags.gpu_devices = next;
                else delete this.flags.gpu_devices;
                this.graph?.afterChange();
                this.setDirtyCanvas(true, true);
            };
            return [
                ...getMenuOptions.apply(this, args),
                null,
                {
                    content: `Execution GPUs: ${selected.length ? selected.join(", ") : "Native"}`,
                    submenu: {
                        options: [
                            { content: "Use native device selection", callback: () => change([]) },
                            null,
                            ...devices.map(({ id, name }) => ({
                                content: `${selected.includes(id) ? "Selected: " : ""}${id} — ${name}`,
                                callback: () => change(selected.includes(id)
                                    ? selected.filter((device) => device !== id)
                                    : [...selected, id]),
                            })),
                        ],
                    },
                },
            ];
        };

        const graphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function (...args) {
            const result = await graphToPrompt.apply(this, args);
            // Resolve placement from the serialized workflow, including API exports.
            const groups = (result.workflow.groups ?? [])
                .filter((group) => group.flags?.gpu_devices?.length)
                .sort((a, b) => (a.bounding[2] * a.bounding[3] - b.bounding[2] * b.bounding[3]) || (a.id - b.id));
            for (const node of result.workflow.nodes) {
                const centerX = node.pos[0] + node.size[0] / 2;
                const centerY = node.pos[1] + node.size[1] / 2;
                const group = groups.find(({ bounding: [x, y, width, height] }) =>
                    centerX >= x && centerX <= x + width && centerY >= y && centerY <= y + height);
                for (const [id, output] of Object.entries(result.output)) {
                    if (id !== String(node.id) && !id.startsWith(`${node.id}:`)) continue;
                    if (group) output._meta = { ...output._meta, gpu_devices: [...group.flags.gpu_devices] };
                    else if (output._meta) delete output._meta.gpu_devices;
                }
            }
            return result;
        };
    },
});
