"""
main.py — FastAPI entry point for PixelAI backend.

Run locally:
    uvicorn main:app --reload --port 8000

Run on the GPU server (exposed to internet via Cloudflare Tunnel / ngrok):
    uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables (all optional):
    PIXELAI_CORS_ORIGINS  comma-separated, e.g. "https://myfrontend.com,http://localhost:5500"
    PIXELAI_MAX_DIM       longest-edge image resize cap before processing (default 2048)
    PIXELAI_DATA_DIR      where to store uploads/outputs (default ./data)
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import logging
import time

from routes.classifier import router as classifier_router
from routes.bgremove   import router as bgremove_router
from routes.bgblur     import router as bgblur_router
from routes.upscale    import router as upscale_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pixelai")

app = FastAPI(title="PixelAI Backend", version="1.1.0")

# --- CORS --------------------------------------------------------------
_cors_env = os.getenv("PIXELAI_CORS_ORIGINS", "*").strip()
_origins  = ["*"] if _cors_env == "*" else [o.strip() for o in _cors_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
log.info(f"CORS origins: {_origins}")

# --- Data dirs ---------------------------------------------------------
DATA_DIR = os.getenv("PIXELAI_DATA_DIR", "data")
os.makedirs(f"{DATA_DIR}/uploads", exist_ok=True)
os.makedirs(f"{DATA_DIR}/outputs", exist_ok=True)
app.mount("/uploads", StaticFiles(directory=f"{DATA_DIR}/uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory=f"{DATA_DIR}/outputs"), name="outputs")

# --- Request timing ----------------------------------------------------
@app.middleware("http")
async def time_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    log.info(f'{request.method} {request.url.path} -> {response.status_code} ({dt:.0f}ms)')
    response.headers["X-Response-Time-ms"] = f"{dt:.0f}"
    return response

# --- Routers -----------------------------------------------------------
app.include_router(classifier_router, prefix="/api/classify",  tags=["Classifier"])
app.include_router(bgremove_router,   prefix="/api/remove-bg", tags=["Background Removal"])
app.include_router(bgblur_router,     prefix="/api/blur-bg",   tags=["Background Blur"])
app.include_router(upscale_router,    prefix="/api/upscale",   tags=["Upscaling"])

# --- Global error handler ---------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exc(_req, exc):
    log.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"error": str(exc)})

# --- Endpoints ---------------------------------------------------------
@app.get("/")
def root():
    return {"status": "PixelAI API is running", "version": app.version}

@app.get("/health")
def health():
    info = {"status": "ok", "cuda_available": False, "device": "CPU", "models": {}}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["device"] = torch.cuda.get_device_name(0) if info["cuda_available"] else "CPU"
        info["torch_version"] = torch.__version__
    except Exception as e:
        info["device"] = f"torch error: {e}"

    # Report which optional model pipelines are wired up
    try:
        from services.bgremove import REMBG_AVAILABLE
        info["models"]["rembg"] = REMBG_AVAILABLE
    except Exception:
        info["models"]["rembg"] = False
    try:
        from services.upscale import realesrgan_status
        info["models"]["realesrgan"] = realesrgan_status()
    except Exception:
        info["models"]["realesrgan"] = {"available": False}
    return info
