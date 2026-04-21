"""
services/bgblur.py — Depth-aware background blur (portrait mode).

Uses MiDaS for monocular depth estimation. The foreground mask is derived
by Otsu-thresholding the normalized depth map, then feathered heavily so
edges look natural rather than cut out.

Defaults to MiDaS_small (fast, good enough for ~2k images). Override with
PIXELAI_MIDAS_MODEL env var:
    MiDaS_small  (fast, ~20MB)
    DPT_Hybrid   (balanced)
    DPT_Large    (slow, highest quality)
"""
import os
import logging
import numpy as np
import cv2
from PIL import Image

log = logging.getLogger("pixelai.bgblur")

_MIDAS_MODEL_NAME = os.getenv("PIXELAI_MIDAS_MODEL", "MiDaS_small")
_midas_model     = None
_midas_transform = None
_device          = None
_midas_loaded    = False


def _load_midas():
    """Lazy-load MiDaS. Sets _midas_model=None if anything fails."""
    global _midas_model, _midas_transform, _device, _midas_loaded
    if _midas_loaded:
        return
    _midas_loaded = True
    try:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _midas_model = torch.hub.load("intel-isl/MiDaS", _MIDAS_MODEL_NAME)
        _midas_model.to(_device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if _MIDAS_MODEL_NAME == "MiDaS_small":
            _midas_transform = transforms.small_transform
        elif _MIDAS_MODEL_NAME == "DPT_Large":
            _midas_transform = transforms.dpt_transform
        else:  # DPT_Hybrid et al
            _midas_transform = transforms.dpt_transform
        log.info(f"MiDaS loaded: {_MIDAS_MODEL_NAME} on {_device}")
    except Exception as e:
        log.warning(f"MiDaS not available: {e} — using radial fallback")
        _midas_model = None


def blur_background(
    pil_image: Image.Image,
    blur_radius: int = 15,
    *,
    fg_softness: int = 31,
) -> Image.Image:
    """
    Depth-aware background blur. Higher blur_radius = more blur on the bg.
    fg_softness is the Gaussian kernel size used to feather the mask.
    """
    blur_radius = max(1, min(60, int(blur_radius)))
    fg_softness = max(3, int(fg_softness) | 1)  # odd, >=3

    img_np = np.array(pil_image.convert("RGB"))
    h, w   = img_np.shape[:2]

    depth = _get_depth(img_np)                                   # (H, W) float
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Otsu picks a per-image threshold rather than a fixed 0.45. This avoids
    # over/under-including the subject for images with different depth ranges.
    d8 = (depth * 255).astype(np.uint8)
    _, fg_bin = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (fg_bin.astype(np.float32) / 255.0)

    # Feather mask heavily so edges blend
    fg = cv2.GaussianBlur(fg, (fg_softness, fg_softness), 0)

    # Blur the background
    ksize   = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(img_np, (ksize, ksize), 0)

    fg3 = np.stack([fg] * 3, axis=2)
    out = (img_np * fg3 + blurred * (1.0 - fg3)).astype(np.uint8)
    return Image.fromarray(out)


def _get_depth(img_np: np.ndarray) -> np.ndarray:
    _load_midas()
    if _midas_model is not None:
        try:
            import torch
            t = _midas_transform(img_np).to(_device)
            with torch.no_grad():
                pred = _midas_model(t)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=img_np.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            return pred.cpu().numpy()
        except Exception as e:
            log.warning(f"MiDaS inference failed: {e} — using radial fallback")

    # Radial fallback: center = foreground
    h, w   = img_np.shape[:2]
    cy, cx = h / 2, w / 2
    y, x   = np.ogrid[:h, :w]
    dist   = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return 1.0 - dist / dist.max()
