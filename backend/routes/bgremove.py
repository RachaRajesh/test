from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from services.bgremove import remove_background
from utils.image_io import save_upload, load_pil_image, save_output
from . import validate_upload

router = APIRouter()

@router.post("/")
async def remove_bg(
    file: UploadFile = File(...),
    alpha_matting: bool = Form(default=True),
):
    validate_upload(file)
    path   = await save_upload(file)
    img    = load_pil_image(path)
    result = remove_background(img, alpha_matting=alpha_matting)
    out    = save_output(result, suffix="_no_bg", fmt="PNG")
    return FileResponse(out, media_type="image/png", filename="removed_bg.png")
