import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

let devices = [];
let enabled = false;

function selectedDevices(group) {
    return group.flags?.gpu_devices ?? devices.map(({ id }) => id);
}

app.registerExtension({
    name: "Comfy.Async.GPUGroups",
    getCanvasMenuItems(canvas) {
        if (!enabled) return [];
        const group = canvas.graph?.getGroupOnPos(...canvas.graph_mouse);
        if (!group) return [];
        return [{
            content: "Execution GPUs",
            has_submenu: true,
            callback: (_item, _options, event, parentMenu) => {
                const items = devices.map(({ id, name }) => ({
                    content: `${id} — ${name}`,
                    callback() {
                        const selected = selectedDevices(group);
                        const next = selected.includes(id)
                            ? selected.filter((device) => device !== id)
                            : [...selected, id];
                        group.graph?.beforeChange();
                        group.flags.gpu_devices = devices.map(({ id }) => id).filter((device) => next.includes(device));
                        group.graph?.afterChange();
                        group.setDirtyCanvas(true, true);
                        update(this, id);
                        return true;
                    },
                }));
                const menu = new LiteGraph.ContextMenu(items, { event, parentMenu });
                for (const [index, entry] of [...menu.root.children].entries()) {
                    const { id } = devices[index];
                    const checkbox = document.createElement("input");
                    checkbox.type = "checkbox";
                    checkbox.tabIndex = -1;
                    checkbox.style.pointerEvents = "none";
                    checkbox.setAttribute("aria-hidden", "true");
                    entry.prepend(checkbox, document.createTextNode(" "));
                    entry.setAttribute("role", "menuitemcheckbox");
                    entry.tabIndex = 0;
                    entry.addEventListener("keydown", (event) => {
                        if (event.key === " " || event.key === "Enter") {
                            event.preventDefault();
                            event.stopPropagation();
                            entry.click();
                        } else if (event.key === "Escape") {
                            event.stopPropagation();
                            menu.close();
                        }
                    });
                    update(entry, id);
                }
                function update(entry, id) {
                    const checked = selectedDevices(group).includes(id);
                    entry.querySelector("input").checked = checked;
                    entry.setAttribute("aria-checked", String(checked));
                }
                return true;
            },
        }];
    },
    async setup() {
        const response = await api.fetchApi("/async/devices");
        if (!response.ok) return;
        ({ enabled, devices } = await response.json());
        if (!enabled) return;

        const draw = LiteGraph.LGraphGroup.prototype.draw;
        LiteGraph.LGraphGroup.prototype.draw = function (canvas, ctx) {
            draw.call(this, canvas, ctx);
            const selected = selectedDevices(this);
            const label = selected.length ? `cuda ${selected.map((id) => id.split(":")[1]).join(",")}` : "no GPUs";
            ctx.save();
            ctx.beginPath();
            ctx.rect(this.pos[0], this.pos[1], this.size[0], this.titleHeight);
            ctx.clip();
            ctx.font = `${LiteGraph.GROUP_TEXT_SIZE}px ${LiteGraph.GROUP_FONT}`;
            ctx.textBaseline = "middle";
            ctx.fillStyle = this._titleTextColor;
            const titleWidth = ctx.measureText(this.title + (this.pinned ? "📌" : "")).width;
            ctx.font = `${LiteGraph.NODE_TEXT_SIZE}px ${LiteGraph.GROUP_FONT}`;
            ctx.fillText(` - ${label}`, this.pos[0] + LiteGraph.GROUP_TEXT_SIZE / 2 + titleWidth,
                this.pos[1] + LiteGraph.NODE_TITLE_HEIGHT / 2 + 1);
            ctx.restore();
        };

        const graphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function (...args) {
            const result = await graphToPrompt.apply(this, args);
            const groups = [...(result.workflow.groups ?? [])]
                .sort((a, b) => (a.bounding[2] * a.bounding[3] - b.bounding[2] * b.bounding[3]) || (a.id - b.id));
            for (const node of result.workflow.nodes) {
                const centerX = node.pos[0] + node.size[0] / 2;
                const centerY = node.pos[1] + node.size[1] / 2;
                const group = groups.find(({ bounding: [x, y, width, height] }) =>
                    centerX >= x && centerX <= x + width && centerY >= y && centerY <= y + height);
                for (const [id, output] of Object.entries(result.output)) {
                    if (id !== String(node.id) && !id.startsWith(`${node.id}:`)) continue;
                    if (group) {
                        const selected = selectedDevices(group);
                        if (!selected.length) throw new Error(`Group "${group.title}" has no GPUs enabled. Select at least one in Execution GPUs.`);
                        output._meta = { ...output._meta, gpu_devices: [...selected] };
                    } else if (output._meta) delete output._meta.gpu_devices;
                }
            }
            return result;
        };
    },
});
