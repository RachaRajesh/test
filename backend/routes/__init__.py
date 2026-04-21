from fastapi import UploadFile, HTTPException

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

def validate_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {sorted(ALLOWED_MIME)}",
        )
