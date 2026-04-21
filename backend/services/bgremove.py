"""
services/bgremove.py — Background removal.

Default model: isnet-general-use (sharper edges than u2net on people/objects).
Override with PIXELAI_REMBG_MODEL env var. Options:
    u2net, u2netp, u2net_human_seg, isnet-general-use, isnet-anime, birefnet-general
"""
import os
import logging
from PIL import Image

log = logging.getLogger("pixelai.bgremove")

_REMBG_MODEL_NAME = os.getenv("PIXELAI_REMBG_MODEL", "isnet-general-use")
_SESSION = None
REMBG_AVAILABLE = False

try:
    from rembg import remove as rembg_remove, new_session
    _SESSION = new_session(_REMBG_MODEL_NAME)
    REMBG_AVAILABLE = True
    log.info(f"rembg loaded with model={_REMBG_MODEL_NAME}")
except Exception as e:
    log.warning(f"rembg not available: {e} — falling back to corner-color heuristic")


def remove_background(pil_image: Image.Image, *, alpha_matting: bool = True) -> Image.Image:
    """
    Remove background. If rembg is installed, uses the configured model with
    alpha matting for softer edges. Falls back to a simple corner-color cut.
    """
    if REMBG_AVAILABLE:
        kwargs = {"session": _SESSION}
        if alpha_matting:
            kwargs.update({
                "alpha_matting": True,
                "alpha_matting_foreground_threshold": 240,
                "alpha_matting_background_threshold": 10,
                "alpha_matting_erode_size": 5,
            })
        try:
            return rembg_remove(pil_image, **kwargs)
        except Exception as e:
            # Alpha matting can fail on tiny/edge-case inputs; retry without.
            log.warning(f"rembg with alpha matting failed ({e}); retrying plain")
            return rembg_remove(pil_image, session=_SESSION)
    return _fallback(pil_image)


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
