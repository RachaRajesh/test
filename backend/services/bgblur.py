"""
services/bgblur.py — Portrait-mode depth-aware background blur.

Strategy: use rembg's subject mask as the foreground anchor (this is what
makes the edges look clean), then optionally use MiDaS depth to add a soft
falloff to the blur strength for objects at mid-distance. This gives
results much closer to phone portrait mode than raw Otsu-on-depth, which
tends to leave artifacts around hair and clothing.

Env vars:
    PIXELAI_BLUR_USE_REMBG   "1" (default) to use rembg for the mask.
                              "0" to fall back to MiDaS + Otsu.
    PIXELAI_MIDAS_MODEL      "MiDaS_small" (fast) | "DPT_Hybrid" | "DPT_Large"
"""
import os
import logging
import numpy as np
import cv2
from PIL import Image

log = logging.getLogger("pixelai.bgblur")

_USE_REMBG_MASK  = os.getenv("PIXELAI_BLUR_USE_REMBG", "1") == "1"
_MIDAS_MODEL_NAME = os.getenv("PIXELAI_MIDAS_MODEL", "MiDaS_small")

_midas_model     = None
_midas_transform = None
_midas_device    = None
_midas_loaded    = False


def _load_midas():
    global _midas_model, _midas_transform, _midas_device, _midas_loaded
    if _midas_loaded:
        return
    _midas_loaded = True
    try:
        import torch
        _midas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _midas_model = torch.hub.load("intel-isl/MiDaS", _MIDAS_MODEL_NAME)
        _midas_model.to(_midas_device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if _MIDAS_MODEL_NAME == "MiDaS_small":
            _midas_transform = transforms.small_transform
        else:
            _midas_transform = transforms.dpt_transform
        log.info(f"MiDaS loaded: {_MIDAS_MODEL_NAME} on {_midas_device}")
    except Exception as e:
        log.warning(f"MiDaS not available: {e}")
        _midas_model = None


def _get_subject_mask_rembg(pil_image: Image.Image) -> np.ndarray:
    """Return a soft (0-1) foreground mask using rembg's segmentation model."""
    from services.bgremove import _load_session
    session = _load_session()
    if session is None:
        return None
    from rembg import remove as rembg_remove
    # Return only the mask, no alpha matting (we feather it ourselves).
    rgba = rembg_remove(pil_image, session=session, only_mask=False, post_process_mask=True)
    alpha = np.array(rgba.split()[-1], dtype=np.float32) / 255.0  # (H, W)
    return alpha


def _get_midas_depth(img_np: np.ndarray) -> np.ndarray:
    _load_midas()
    if _midas_model is None:
        return None
    try:
        import torch
        t = _midas_transform(img_np).to(_midas_device)
        with torch.no_grad():
            pred = _midas_model(t)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = pred.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth
    except Exception as e:
        log.warning(f"MiDaS inference failed: {e}")
        return None


def blur_background(
    pil_image: Image.Image,
    blur_radius: int = 15,
    *,
    feather: int = 15,
) -> Image.Image:
    """
    Depth-aware background blur.

    blur_radius — maximum blur strength at the far background (1-60).
    feather     — odd number, how soft the foreground/background transition is.
    """
    blur_radius = max(1, min(60, int(blur_radius)))
    feather     = max(3, int(feather) | 1)

    img_np = np.array(pil_image.convert("RGB"))
    h, w   = img_np.shape[:2]

    # --- Foreground mask ---
    fg = None
    if _USE_REMBG_MASK:
        fg = _get_subject_mask_rembg(pil_image)

    if fg is None:
        # Fallback: MiDaS + Otsu
        depth = _get_midas_depth(img_np)
        if depth is not None:
            d8 = (depth * 255).astype(np.uint8)
            _, fg_bin = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            fg = fg_bin.astype(np.float32) / 255.0
        else:
            # Pure radial fallback (no models available)
            cy, cx = h / 2, w / 2
            y, x   = np.ogrid[:h, :w]
            dist   = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            fg = (1.0 - dist / dist.max()).astype(np.float32)

    # --- Feather the mask for smooth transitions ---
    fg = cv2.GaussianBlur(fg, (feather, feather), 0)
    fg = np.clip(fg, 0.0, 1.0)

    # --- Blur the background ---
    ksize   = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(img_np, (ksize, ksize), 0)

    fg3 = np.stack([fg] * 3, axis=2)
    out = (img_np * fg3 + blurred * (1.0 - fg3)).astype(np.uint8)
    return Image.fromarray(out)
