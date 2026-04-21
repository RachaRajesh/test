"""
services/upscale.py — Real-ESRGAN AI upscaling (x2 / x4).

Weight files must exist at backend/models/weights/:
    RealESRGAN_x2plus.pth
    RealESRGAN_x4plus.pth

If basicsr + realesrgan + weights are not available, falls back to PIL LANCZOS
and surfaces the fallback in the response headers via realesrgan_status().
"""
import os
import logging
import traceback
import numpy as np
from PIL import Image

log = logging.getLogger("pixelai.upscale")

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "weights")
_WEIGHTS = {
    2: "RealESRGAN_x2plus.pth",
    4: "RealESRGAN_x4plus.pth",
}
_models = {}
_last_error = None


def realesrgan_status() -> dict:
    """Summary of which scales are available — used by /health."""
    status = {"available": {}, "weights_dir": os.path.abspath(WEIGHTS_DIR), "last_error": _last_error}
    for scale, fname in _WEIGHTS.items():
        status["available"][f"x{scale}"] = os.path.exists(os.path.join(WEIGHTS_DIR, fname))
    return status


def _load_esrgan(scale: int):
    """Lazy-load ESRGAN upsampler for a given scale. Returns None on failure."""
    global _last_error
    if scale in _models:
        return _models[scale]

    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        weight_path = os.path.abspath(os.path.join(WEIGHTS_DIR, _WEIGHTS[scale]))
        if not os.path.exists(weight_path):
            _last_error = f"Weights missing: {weight_path}"
            log.warning(_last_error)
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=scale,
        )
        upsampler = RealESRGANer(
            scale=scale,
            model_path=weight_path,
            model=net,
            tile=512,       # tiling avoids OOM on big images
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            device=device,
        )
        _models[scale] = upsampler
        log.info(f"Real-ESRGAN x{scale} loaded on {device}")
        return upsampler

    except Exception as e:
        _last_error = repr(e)
        log.warning(f"Real-ESRGAN x{scale} unavailable: {e}")
        log.debug(traceback.format_exc())
        return None


def upscale_image(pil_image: Image.Image, scale: int = 2) -> tuple[Image.Image, str]:
    """
    Returns (upscaled_image, method) where method is "realesrgan" or "lanczos".
    """
    if scale not in _WEIGHTS:
        raise ValueError(f"Scale must be one of {list(_WEIGHTS)}; got {scale}")

    upsampler = _load_esrgan(scale)
    if upsampler is not None:
        import cv2
        bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        out, _ = upsampler.enhance(bgr, outscale=scale)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), "realesrgan"

    log.info(f"Using PIL LANCZOS fallback x{scale}")
    w, h = pil_image.size
    return pil_image.resize((w * scale, h * scale), Image.LANCZOS), "lanczos"
