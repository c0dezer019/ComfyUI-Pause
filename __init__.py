import server
from aiohttp import web
from .pause_nodes import PSampler, Signaler
from .shared import PAUSE_STATE, state_lock

# --- API Route Registration ---
@server.PromptServer.instance.routes.post("/comfy/steer")
async def set_steering_command(request):
    """
    Receives control signals from the frontend (The Cockpit).
    Payload example: { "command": "PAUSE" }
    """
    json_data = await request.json()
    command = json_data.get("command", "PROCEED")
    
    # Update the global state
    with state_lock:
        PAUSE_STATE["command"] = command
    
    print(f"[ComfyUI-Pause] 📡 Signal Received: {command}")
    return web.json_response({"status": "success", "received": command})

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "PSampler": PSampler,
    "Signaler": Signaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSampler": "⏸️ P-Sampler (Pause Enabled)",
    "Signaler": "🚦 Signaler (Control)"
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "PAUSE_STATE", "WEB_DIRECTORY"]
