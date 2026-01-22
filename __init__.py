import server
from aiohttp import web
from .pause_nodes import PSampler, PSamplerAdvanced
from .shared import PAUSE_STATE, state_lock

# API Route
@server.PromptServer.instance.routes.post("/comfy/pause_signal")
async def set_pause_command(request):
    json_data = await request.json()
    command = json_data.get("command", "PROCEED")
    
    with state_lock:
        PAUSE_STATE["command"] = command
    
    print(f"[P-Sampler] 📡 Signal Received: {command}")
    return web.json_response({"status": "success"})

# Node Mappings
NODE_CLASS_MAPPINGS = {
    "PSampler": PSampler,
    "PSamplerAdvanced": PSamplerAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSampler": "⏸️ Pausable Sampler",
    "PSamplerAdvanced": "⏸️ Pausable Sampler (Advanced)"
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
