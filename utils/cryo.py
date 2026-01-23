import torch
import json
from safetensors.torch import save_file, load_file

class CryoStateManager:
    @staticmethod
    def save_state(path, latent, step, settings, derivative=None):
        """Saves a 'Math Slice' containing latent, RNG, and momentum data."""
        tensors = {
            "latent": latent.cpu(), [cite: 29]
            "rng_cpu": torch.get_rng_state() [cite: 16]
        }
        
        if torch.cuda.is_available():
            tensors["rng_cuda"] = torch.cuda.get_rng_state().cpu() [cite: 29]
        
        if derivative is not None:
            tensors["derivative"] = derivative.cpu() [cite: 30]

        metadata = {
            "step": str(step), [cite: 7]
            "settings": json.dumps(settings), [cite: 7]
            "has_momentum": "True" if derivative is not None else "False"
        }
        
        save_file(tensors, path, metadata=metadata) [cite: 7, 26]

    @staticmethod
    def load_state(path):
        """Loads a slice and restores the RNG environment."""
        data = load_file(path) [cite: 21]
        
        # Restore RNG for mathematical continuity
        torch.set_rng_state(data["rng_cpu"]) [cite: 16]
        if "rng_cuda" in data and torch.cuda.is_available():
            torch.cuda.set_rng_state(data["rng_cuda"]) [cite: 16]
            
        return data
