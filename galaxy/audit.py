"""audit_log.py

Lightweight audit logging utilities.

This module is intentionally dependency-light and safe to import from both GUI
and backend threads.

It provides:
- QtLogHandler: emits log lines to a Qt signal and optionally mirrors to file.
- save_run_config: write a JSON with all parameters + ROI vertices for reproducibility.

License: MIT
"""

from __future__ import annotations

import json
import os
import sys
import datetime as _dt
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from qtpy import QtCore


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    # numpy / pandas types
    try:
        import numpy as np

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass  # numpy not available; fall through to str() conversion

    # fallback
    return str(obj)


class QtLogHandler(QtCore.QObject):
    """Logger that sends messages to a Qt signal and optionally mirrors to a file."""

    sig = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._file_handle = None

    def set_log_file(self, path: Optional[str]) -> None:
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass  # close() failed (e.g. already closed); continue to reset handle
            self._file_handle = None

        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            try:
                self._file_handle = open(path, "a", encoding="utf-8")
            except OSError as exc:
                # Log to stderr only — we cannot use the logger here (it calls us).
                import sys
                print(f"[GalaXY] WARNING: Could not open log file {path!r}: {exc}", file=sys.stderr)

    def _emit(self, level: str, msg: str) -> None:
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {level}: {msg}"
        self.sig.emit(line)
        if self._file_handle:
            try:
                self._file_handle.write(line + "\n")
                self._file_handle.flush()
            except Exception:
                # avoid crashing the GUI due to IO issues
                pass

    def info(self, msg: str) -> None:
        self._emit("INFO", msg)

    def warning(self, msg: str) -> None:
        self._emit("WARNING", msg)

    def error(self, msg: str) -> None:
        self._emit("ERROR", msg)


def save_run_config(path: str, config: Dict[str, Any]) -> None:
    """Write JSON config for full audit trail."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cfg = dict(config)
    cfg.setdefault("created_at", _dt.datetime.now().isoformat())
    cfg.setdefault("python", sys.version)

    # Try to record versions of key packages.
    versions = {}
    for pkg in ("numpy", "scipy", "pandas", "shapely", "skimage", "networkx", "sklearn"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            versions[pkg] = None
    cfg.setdefault("package_versions", versions)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(cfg), f, indent=2)
