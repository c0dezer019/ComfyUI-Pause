import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.Pause.Signaler",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Signaler") {
            console.log("[ComfyUI-Pause] ✅ Found Signaler node definition! Injecting buttons...");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // 1. Add the Pause Button
                this.addWidget("button", "🛑 PAUSE", null, () => {
                    sendBridgeSignal("PAUSE");
                });

                // 2. Add the Resume Button
                this.addWidget("button", "▶️ RESUME", null, () => {
                    sendBridgeSignal("PROCEED");
                });

                // 3. Force the node to be large enough to show buttons
                this.setSize([200, 100]);

                console.log("[ComfyUI-Pause] 🛠️ Buttons added to new Signaler node.");
                return r;
            };
        }
    }
});

async function sendBridgeSignal(command) {
    console.log(`[ComfyUI-Pause] 📤 Sending: ${command}`);
    try {
        const response = await api.fetchApi("/comfy/steer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: command }),
        });
    } catch (error) {
        console.error("[ComfyUI-Pause] ❌ API Error:", error);
    }
}
