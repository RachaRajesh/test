from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from services.upscale import upscale_image
from utils.image_io import save_upload, load_pil_image, save_output
from . import validate_upload

router = APIRouter()

@router.post("/")
async def upscale(
    file: UploadFile = File(...),
    scale: int = Form(default=4),
):
    validate_upload(file)
    if scale not in (2, 4):
        raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
    path           = await save_upload(file)
    img            = load_pil_image(path)
    result, method = upscale_image(img, scale=scale)
    out            = save_output(result, suffix=f"_x{scale}", fmt="PNG")
    return FileResponse(
        out,
        media_type="image/png",
        filename=f"upscaled_x{scale}.png",
        headers={"X-Upscale-Method": method},
    )
