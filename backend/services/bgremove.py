"""
services/bgremove.py — Background removal with BiRefNet (best quality).

Default model: birefnet-general — state-of-the-art subject segmentation,
dramatically cleaner edges than U2Net or ISNet.

Override with PIXELAI_REMBG_MODEL env var. Options (fastest -> best):
    u2netp (fast, tiny)
    u2net (classic)
    isnet-general-use (sharper than u2net)
    birefnet-general (DEFAULT — best quality)
    birefnet-general-lite (same architecture, half the size)
    birefnet-portrait (portraits only — use this if photo is just a person)

Alpha matting is enabled by default for soft hair/fur edges.
"""
import os
import logging
from PIL import Image

log = logging.getLogger("pixelai.bgremove")

_REMBG_MODEL_NAME = os.getenv("PIXELAI_REMBG_MODEL", "birefnet-general")
_SESSION = None
REMBG_AVAILABLE = False
_LOAD_ERROR = None


def _load_session():
    """Lazy-load rembg session so startup isn't blocked by model download."""
    global _SESSION, REMBG_AVAILABLE, _LOAD_ERROR
    if _SESSION is not None:
        return _SESSION
    try:
        from rembg import new_session
        _SESSION = new_session(_REMBG_MODEL_NAME)
        REMBG_AVAILABLE = True
        log.info(f"rembg loaded with model={_REMBG_MODEL_NAME}")
        return _SESSION
    except Exception as e:
        _LOAD_ERROR = repr(e)
        log.error(f"rembg failed to load model '{_REMBG_MODEL_NAME}': {e}")
        return None


def remove_background(pil_image: Image.Image, *, alpha_matting: bool = True) -> Image.Image:
    """
    Remove background with BiRefNet (default). Alpha matting on by default for
    softer, more natural edges around hair and fine details.
    """
    session = _load_session()
    if session is None:
        log.warning("Using naive corner-color fallback (rembg not loaded)")
        return _fallback(pil_image)

    from rembg import remove as rembg_remove
    kwargs = {"session": session}
    if alpha_matting:
        # Tuned values for BiRefNet. Foreground threshold = confidence needed
        # to treat a pixel as foreground; background threshold the inverse.
        # Erode size controls how much the binary mask shrinks before matting
        # (smaller = preserve fine details like hair).
        kwargs.update({
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 270,
            "alpha_matting_background_threshold": 20,
            "alpha_matting_erode_size": 11,
        })
    try:
        return rembg_remove(pil_image, **kwargs)
    except Exception as e:
        # Alpha matting can fail on tiny/edge-case inputs; retry without.
        log.warning(f"rembg with alpha matting failed ({e}); retrying plain")
        return rembg_remove(pil_image, session=session)


def model_info() -> dict:
    return {
        "name": _REMBG_MODEL_NAME,
        "loaded": REMBG_AVAILABLE,
        "error": _LOAD_ERROR,
    }


def _fallback(pil_image: Image.Image) -> Image.Image:
    """Naive fallback: treat average of 4 corners as background, cut it out."""
    import numpy as np
    img  = pil_image.convert("RGBA")
    data = np.array(img, dtype=float)
    corners = [data[0, 0, :3], data[0, -1, :3], data[-1, 0, :3], data[-1, -1, :3]]
    bg   = np.mean(corners, axis=0)
    diff = np.sqrt(((data[:, :, :3] - bg) ** 2).sum(axis=2))
    data[diff < 60, 3] = 0
    return Image.fromarray(data.astype(np.uint8), "RGBA")
