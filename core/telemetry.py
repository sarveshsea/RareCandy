import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

class TelemetryWriter:
    """
    Writes bot state to a JSON file and appends to a log file.
    Designed to be served via simple HTTP for remote monitoring (Wurmple).
    """
    def __init__(self, output_dir: str = "monitor"):
        self.output_dir = output_dir
        self.status_file = os.path.join(output_dir, "status.json")
        self.log_file = os.path.join(output_dir, "events.log")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def update_state(self, equity: float, positions: Dict[str, float], active_signal: Optional[str] = None):
        """Update the heartbeat status.json"""
        state = {
            "timestamp": time.time(),
            "last_update": datetime.now().isoformat(),
            "equity": equity,
            "positions": positions,
            "active_signal": active_signal,
            "healthy": True
        }
        
        try:
            # Atomic write (write temp then rename) to prevent read errors
            temp_file = self.status_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(temp_file, self.status_file)
        except Exception as e:
            print(f"[Telemetry] Write Error: {e}")

    def log_event(self, event_type: str, message: str):
        """Append a structured event line."""
        timestamp = datetime.now().isoformat()
        line = f"[{timestamp}] [{event_type}] {message}\n"
        try:
            with open(self.log_file, "a") as f:
                f.write(line)
        except Exception as e:
            print(f"[Telemetry] Log Error: {e}")
