"""
services/bgremove.py — Background removal with BiRefNet (best quality).

Default model: birefnet-general — state-of-the-art, dramatically cleaner
edges than U2Net or ISNet.

Override with PIXELAI_REMBG_MODEL env var:
    u2netp / u2net / u2net_human_seg
    isnet-general-use / isnet-anime
    birefnet-general (DEFAULT)
    birefnet-general-lite
    birefnet-portrait

IMPORTANT: if the model fails to load, this service raises instead of
silently falling back to a naive cutter. That way the frontend sees a clear
error and you can diagnose rather than ship a bad result.
"""
import os
import logging
from PIL import Image

log = logging.getLogger("pixelai.bgremove")

_REMBG_MODEL_NAME = os.getenv("PIXELAI_REMBG_MODEL", "birefnet-general")
_ALLOW_NAIVE_FALLBACK = os.getenv("PIXELAI_ALLOW_NAIVE_BG_FALLBACK", "0") == "1"

_SESSION = None
REMBG_AVAILABLE = False
_LOAD_ERROR = None


class BackgroundRemovalUnavailable(Exception):
    """Raised when rembg/onnxruntime isn't installed or the model won't load."""


def _load_session():
    """Lazy-load rembg session so startup isn't blocked by model download."""
    global _SESSION, REMBG_AVAILABLE, _LOAD_ERROR
    if _SESSION is not None:
        return _SESSION
    try:
        from rembg import new_session
    except ImportError as e:
        _LOAD_ERROR = f"rembg not installed: {e}. Run: pip install 'rembg[gpu]'"
        log.error(_LOAD_ERROR)
        return None

    try:
        _SESSION = new_session(_REMBG_MODEL_NAME)
        REMBG_AVAILABLE = True
        log.info(f"rembg session created with model={_REMBG_MODEL_NAME}")
        return _SESSION
    except Exception as e:
        _LOAD_ERROR = f"Failed to create session for '{_REMBG_MODEL_NAME}': {e!r}"
        log.error(_LOAD_ERROR)
        return None


def remove_background(pil_image: Image.Image, *, alpha_matting: bool = True) -> Image.Image:
    """
    Remove background with BiRefNet. Alpha matting ON for soft hair edges.

    Raises BackgroundRemovalUnavailable if the model can't run.
    Set PIXELAI_ALLOW_NAIVE_BG_FALLBACK=1 to fall back to the toy corner-color
    cutter (NOT recommended — it produces jagged, hole-ridden masks).
    """
    session = _load_session()
    if session is None:
        if _ALLOW_NAIVE_FALLBACK:
            log.warning("Using naive fallback (rembg not loaded). Enable rembg for real results.")
            return _fallback(pil_image)
        raise BackgroundRemovalUnavailable(
            _LOAD_ERROR or "rembg session not available"
        )

    from rembg import remove as rembg_remove
    kwargs = {"session": session}
    if alpha_matting:
        kwargs.update({
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 270,
            "alpha_matting_background_threshold": 20,
            "alpha_matting_erode_size": 11,
        })
    try:
        return rembg_remove(pil_image, **kwargs)
    except Exception as e:
        log.warning(f"Alpha matting failed ({e}); retrying without.")
        return rembg_remove(pil_image, session=session)


def model_info() -> dict:
    # Trigger a load attempt so health reflects real state, not "not yet tried"
    _load_session()
    return {
        "name": _REMBG_MODEL_NAME,
        "loaded": REMBG_AVAILABLE,
        "error": _LOAD_ERROR,
    }


def _fallback(pil_image: Image.Image) -> Image.Image:
    """Toy fallback — DO NOT use in production. Cuts out corner color."""
    import numpy as np
    img  = pil_image.convert("RGBA")
    data = np.array(img, dtype=float)
    corners = [data[0, 0, :3], data[0, -1, :3], data[-1, 0, :3], data[-1, -1, :3]]
    bg   = np.mean(corners, axis=0)
    diff = np.sqrt(((data[:, :, :3] - bg) ** 2).sum(axis=2))
    data[diff < 60, 3] = 0
    return Image.fromarray(data.astype(np.uint8), "RGBA")
