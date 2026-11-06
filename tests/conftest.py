"""Test setup: make LiteDicom.py importable on machines without a display / Tk.

LiteDicom.py is a single-file app that imports Tkinter at module level and exits
if it is missing. The tests only exercise the GUI-free helpers, so when Tk (or
Pillow's ImageTk) can't be imported we register harmless stand-ins instead.
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import tkinter  # noqa: F401
    from PIL import ImageTk  # noqa: F401
except Exception:
    for name in ("tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
                 "tkinter.colorchooser", "PIL.ImageTk"):
        sys.modules[name] = mock.MagicMock(name=name)
