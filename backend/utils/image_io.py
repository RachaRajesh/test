import os
import uuid
from pathlib import Path
from PIL import Image
from fastapi import UploadFile

DATA_DIR   = Path(os.getenv("PIXELAI_DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
MAX_DIM    = int(os.getenv("PIXELAI_MAX_DIM", "2048"))


async def save_upload(file: UploadFile) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext  = Path(file.filename or "").suffix.lower() or ".jpg"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    contents = await file.read()
    dest.write_bytes(contents)
    return dest


def load_pil_image(path: Path, max_dim: int = MAX_DIM) -> Image.Image:
    img = Image.open(path)
    # Respect EXIF rotation (fixes sideways phone photos)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def save_output(pil_image: Image.Image, suffix: str = "_out", fmt: str = "PNG") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ext  = ".png" if fmt.upper() == "PNG" else ".jpg"
    dest = OUTPUT_DIR / f"{uuid.uuid4().hex}{suffix}{ext}"
    kw   = {}
    if fmt.upper() == "JPEG":
        kw["quality"] = 92
        pil_image = pil_image.convert("RGB")
    pil_image.save(dest, format=fmt, **kw)
    return dest
