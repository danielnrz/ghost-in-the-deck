"""One shared Panda3D instance for the whole test run.

Panda3D expects a single ShowBase per process, so every test that needs the
runtime goes through here. An offscreen window is used when a display is
available, which gives the render tests a real GL context; otherwise the tests
that need pixels skip and the rest still run.
"""

from __future__ import annotations

import os

_base = None
_has_window = False


def get_base():
    global _base, _has_window
    if _base is not None:
        return _base

    from panda3d.core import loadPrcFileData

    want_window = bool(os.environ.get("DISPLAY"))
    loadPrcFileData("", "window-type offscreen" if want_window else "window-type none")
    loadPrcFileData("", "win-size 640 480")
    loadPrcFileData("", "audio-library-name null")

    from direct.showbase.ShowBase import ShowBase

    _base = ShowBase()
    _has_window = want_window and _base.win is not None
    return _base


def has_window() -> bool:
    get_base()
    return _has_window


def render_screenshot():
    """Render one frame offscreen and return it as an (h, w, 3) uint8 array."""
    import numpy as np

    base = get_base()
    base.graphicsEngine.renderFrame()
    base.graphicsEngine.renderFrame()

    texture = base.win.getScreenshot()
    if texture is None:
        raise RuntimeError("could not capture a frame")
    raw = texture.getRamImageAs("RGB")
    if not raw:
        raise RuntimeError("captured frame has no data")
    pixels = np.frombuffer(bytes(raw), dtype=np.uint8)
    return pixels.reshape(texture.getYSize(), texture.getXSize(), 3)
