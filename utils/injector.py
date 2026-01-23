import torch
from pathlib import Path
from .cryo import CryoStateManager

class HistoryInjector:
    @staticmethod
    def gather_history(run_dir: Path, current_step: int, depth: int = 3):
        """
        Walks backwards from current_step to find previous derivatives.
        Returns a list of tensors sorted chronologically: [d_t-3, d_t-2, d_t-1]
        """
        history_buffer = []
        
        # We look back 'depth' steps (e.g., for DPM++ 3M we need 3 previous steps)
        for i in range(1, depth + 1):
            target_step = current_step - i
            if target_step < 1:
                break # We hit the start of the generation
            
            # Construct expected filename from the piecemeal log
            filename = f"step_{target_step}.safetensors"
            file_path = run_dir / filename
            
            if not file_path.exists():
                print(f"⚠️ PSampler Injection Warning: Missing history file {filename}. Momentum will be imperfect.")
                continue
                
            # Load just the derivative tensor (we don't need the full latent/RNG here)
            # We assume load_state returns the dict with 'derivative' key
            try:
                state = CryoStateManager.load_state(file_path)
                if "derivative" in state:
                    # Insert at the beginning to maintain chronological order [Oldest -> Newest]
                    history_buffer.insert(0, state["derivative"])
                else:
                    print(f"⚠️ PSampler: File {filename} exists but has no momentum data.")
            except Exception as e:
                print(f"❌ PSampler Read Error: {e}")

        print(f"💉 Injection: Rebuilt history buffer with {len(history_buffer)} frames.")
        return history_buffer
