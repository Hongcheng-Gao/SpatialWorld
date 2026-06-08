"""Console helpers shared by interactive environment scripts."""

from __future__ import annotations

import os
import sys


def configure_utf8_stdio() -> None:
    """Prefer UTF-8 for terminal I/O without failing on redirected streams."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass
