from fastapi import APIRouter, UploadFile, File
from services.classifier import classify_image
from utils.image_io import save_upload, load_pil_image
from . import validate_upload

router = APIRouter()

@router.post("/")
async def classify(file: UploadFile = File(...)):
    validate_upload(file)
    path = await save_upload(file)
    img  = load_pil_image(path)
    return classify_image(img)
