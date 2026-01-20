import threading

# The Lock ensures that the API and the GPU don't fight over this variable
state_lock = threading.Lock()

PAUSE_STATE = {
    "command": "PROCEED",
    "payload": None
}
