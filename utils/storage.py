import os
import shutil
from pathlib import Path

class SnapshotManager:
    def __init__(self, run_id, base_dir, storage_limit_mb=2048):
        self.run_dir = Path(base_dir) / run_id [cite: 31]
        self.run_dir.mkdir(parents=True, exist_ok=True) [cite: 31]
        self.limit_bytes = storage_limit_mb * 1024 * 1024 [cite: 4]
        self.history = [] # Tracks {'step': int, 'path': Path, 'size': int}

    def enforce_limit(self, next_file_size):
        """Deletes oldest snapshots if the new one would exceed the budget."""
        current_usage = sum(f.stat().st_size for f in self.run_dir.glob("*.safetensors")) [cite: 4]
        while (current_usage + next_file_size) > self.limit_bytes and self.history:
            oldest = self.history.pop(0) [cite: 4]
            if oldest['path'].exists():
                current_usage -= oldest['size']
                oldest['path'].unlink() [cite: 4]

    def register(self, path, step):
        self.history.append({'step': step, 'path': path, 'size': path.stat().st_size})
        self.history.sort(key=lambda x: x['step']) # Ensure chronological order [cite: 4]
