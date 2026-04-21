from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from services.bgblur import blur_background
from utils.image_io import save_upload, load_pil_image, save_output
from . import validate_upload

router = APIRouter()

@router.post("/")
async def blur_bg(
    file: UploadFile = File(...),
    blur_radius: int = Form(default=15),
):
    validate_upload(file)
    blur_radius = max(1, min(60, int(blur_radius)))
    path   = await save_upload(file)
    img    = load_pil_image(path)
    result = blur_background(img, blur_radius=blur_radius)
    out    = save_output(result, suffix="_blurred", fmt="JPEG")
    return FileResponse(out, media_type="image/jpeg", filename="blurred_bg.jpg")
