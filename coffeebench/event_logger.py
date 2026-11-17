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

    Checkpoint/resume: pickling keeps only (path, byte offset, elapsed ms);
    the file handle and lock are dropped. `EventLogger(path,
    truncate_to=offset, elapsed_ms=ms)` reopens the file in append mode,
    truncated to the checkpoint offset, so the stream stays continuous and
    `ts_ms` keeps counting from where the checkpoint left it.
    """

    def __init__(
        self,
        path: str,
        truncate_to: int | None = None,
        elapsed_ms: int = 0,
    ) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if truncate_to is None:
            # Truncate on new run so the file represents one run end-to-end.
            self._fh = open(path, "w", buffering=1)  # line-buffered
        else:
            # Resume: drop everything written after the checkpoint.
            with open(path, "ab") as f:
                f.truncate(int(truncate_to))
            self._fh = open(path, "a", buffering=1)
        self._t0 = time.time() - elapsed_ms / 1000.0
        self._lock = threading.Lock()

    # ----- checkpoint support -----

    def offset(self) -> int:
        """Byte offset of the end of the stream (everything flushed)."""
        with self._lock:
            if self._fh.closed:
                return os.path.getsize(self.path)
            self._fh.flush()
            return self._fh.tell()

    def elapsed_ms(self) -> int:
        return int((time.time() - self._t0) * 1000)

    def __getstate__(self) -> dict:
        return {
            "path": self.path,
            "offset": self.offset(),
            "elapsed_ms": self.elapsed_ms(),
        }

    def __setstate__(self, state: dict) -> None:
        # Detached: no file handle until the resume code reopens the stream
        # (see coffeebench.server.checkpoint.restore). emit() is a no-op.
        self.path = state["path"]
        self.checkpoint_offset = state["offset"]
        self.checkpoint_elapsed_ms = state["elapsed_ms"]
        self._fh = None
        self._t0 = time.time() - state["elapsed_ms"] / 1000.0
        self._lock = threading.Lock()

    # ----- emit -----

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
        if self._fh is None:
            return
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
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
