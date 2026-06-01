import json
import os
import threading
import time
from dataclasses import asdict, is_dataclass


class EventLogger:
    """Append-only JSONL event stream for live-monitoring a simulation run.

    Each emitted event is one JSON object per line with a monotonic `ts_ms`
    and a `type` discriminator (`step`, `email`, `week_end`, `run_start`,
    `run_end`). Kept intentionally thin — the web UI replays the file to
    reconstruct state on load, then tails for live updates.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Truncate on new run so the file represents one run end-to-end.
        self._fh = open(path, "w", buffering=1)  # line-buffered
        self._t0 = time.time()
        self._lock = threading.Lock()

    def _jsonable(self, value):
        """Best-effort convert dataclasses / common objects to JSON-safe dicts."""
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {k: self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        return value

    def emit(self, event_type: str, **data) -> None:
        event = {
            "ts_ms": int((time.time() - self._t0) * 1000),
            "type": event_type,
            **self._jsonable(data),
        }
        # Serialize writes so concurrent emits from parallel agent turns don't
        # interleave within a line.
        with self._lock:
            self._fh.write(json.dumps(event, default=str) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
