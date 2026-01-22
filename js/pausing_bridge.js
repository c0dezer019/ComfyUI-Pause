import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.Pause.Control",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Target both Standard and Advanced nodes
        if (nodeData.name === "PSampler" || nodeData.name === "PSamplerAdvanced") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // 1. Control Deck
                this.addWidget("button", "▶️ RESUME", null, () => sendSignal("PROCEED"));
                this.addWidget("button", "⏸️ PAUSE", null, () => sendSignal("PAUSE"));

                // 2. Precision CFG Fix
                const cfgWidget = this.widgets.find(w => w.name === "cfg");
                if (cfgWidget) {
                    cfgWidget.step = 0.1;
                    if (!cfgWidget.options) cfgWidget.options = {};
                    cfgWidget.options.step = 0.1;
                    cfgWidget.options.precision = 1;
                }

                this.setSize([300, nodeData.name === "PSamplerAdvanced" ? 440 : 280]);
                return r;
            };
        }
    }
});

async function sendSignal(command) {
    try {
        await api.fetchApi("/comfy/pause_signal", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: command }),
        });
    } catch (error) {
        console.error("[P-Sampler] API Error:", error);
    }
}
