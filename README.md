# PixelAI — AI-Powered Image Editor

> Capstone Project · Spring 2026 · Kent State University · Group 3

A full-stack image editor that combines classical editing tools with modern AI models — background removal, depth-aware blur, AI upscaling, and photo quality scoring — all accessible from a browser, powered by a FastAPI backend running on an NVIDIA H100 GPU.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+cu124-EE4C2C)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900)

---

## Features

### AI Tools
| Tool | Model | Notes |
|------|-------|-------|
| **Background Removal** | BiRefNet (via rembg) | Alpha-matted output for clean hair/fur edges. PNG with transparency. |
| **Depth-Aware Blur** | BiRefNet mask + Gaussian blur | Portrait-mode style background blur. Adjustable radius 1–60. |
| **AI Upscaling** | Real-ESRGAN x2/x4 | Super-resolution for low-res photos. Falls back to LANCZOS if weights missing. |
| **Photo Quality Classifier** | Classical CV metrics | 1–10 score using sharpness, exposure, contrast, noise, and face detection. |

### Classical Editing
- Brightness, contrast, saturation, blur sliders (live, non-destructive)
- Crop with aspect-ratio lock (free / 1:1 / 4:3 / 16:9)
- 20-step undo/redo history (including slider adjustments)
- Before/after comparison slider after AI operations
- Drag-and-drop upload, PNG export
- Keyboard shortcuts (V, C, Ctrl+Z, Ctrl+Y, Ctrl+S)
- "Reset All" reverts to the original uploaded image

### Deployment
- Single-server setup: FastAPI serves the frontend AND the API from the same URL
- Expose to the internet via Cloudflare Tunnel or ngrok
- Lightweight frontend (three static files — no build step)

---

## Quick Start

### Option A — Local (CPU, dev only)

```bash
git clone https://github.com/siragena/AI-Era-Image-Editor.git
cd AI-Era-Image-Editor/backend

python3 -m venv venv
source venv/bin/activate                    # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install "rembg[cpu]"

uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser. The frontend loads automatically. That's it.

### Option B — GPU Server (full setup with every gotcha we hit)

This is the setup that works on a shared GPU server with a read-only home directory (like Kent State's Viper).

```bash
# 1. Clone into a writable directory (NOT your home dir if it's read-only)
cd /tmp/project
git clone https://github.com/siragena/AI-Era-Image-Editor.git test
cd test/backend

# 2. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 3. Install GPU PyTorch — match your CUDA version (check with `nvidia-smi`)
# CUDA 12.x:
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CUDA 11.8 instead:
# pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Base requirements
pip install -r requirements.txt

# 5. rembg with GPU support
pip install "rembg[gpu]"

# 6. Real-ESRGAN dependencies
pip install basicsr realesrgan

# 7. PATCH basicsr — newer torchvision removed the function it imports.
#    Without this patch, the upscale pipeline crashes on first use.
sed -i 's|torchvision.transforms.functional_tensor|torchvision.transforms.functional|' \
  venv/lib/python3.12/site-packages/basicsr/data/degradations.py

# Verify the patch worked:
python -c "from realesrgan import RealESRGANer; print('OK')"

# 8. Download Real-ESRGAN weights
mkdir -p models/weights
cd models/weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
cd ../..

# 9. Create writable cache dirs (rembg and MiDaS try to cache in ~ by default)
mkdir -p /tmp/project/u2net-cache
mkdir -p /tmp/project/torch-cache

# 10. Start the server with env vars pointing to writable cache paths
export U2NET_HOME=/tmp/project/u2net-cache
export TORCH_HOME=/tmp/project/torch-cache
export XDG_CACHE_HOME=/tmp/project/torch-cache
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://<server-ip>:8000` in your browser.

### Startup script (recommended)

To avoid retyping env vars every time:

```bash
cat > /tmp/project/test/backend/run.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export U2NET_HOME=/tmp/project/u2net-cache
export TORCH_HOME=/tmp/project/torch-cache
export XDG_CACHE_HOME=/tmp/project/torch-cache
mkdir -p "$U2NET_HOME" "$TORCH_HOME"
exec uvicorn main:app --host 0.0.0.0 --port 8000
EOF
chmod +x /tmp/project/test/backend/run.sh
```

Next time just: `./run.sh`

### Verify the server is working

```bash
curl http://localhost:8000/health | python3 -m json.tool
```

You want:
```json
{
  "status": "ok",
  "cuda_available": true,
  "device": "NVIDIA H100 NVL",
  "torch_version": "2.6.0+cu124",
  "models": {
    "rembg": true,
    "realesrgan": {"available": {"x2": true, "x4": true}, "last_error": null}
  }
}
```

---

## How it works — One server, one URL

FastAPI serves BOTH the API and the frontend from the same URL:

| URL | What it returns |
|-----|-----------------|
| `http://localhost:8000/` | The image editor UI (serves `frontend/index.html`) |
| `http://localhost:8000/api/remove-bg/` | Background removal endpoint |
| `http://localhost:8000/api/upscale/` | AI upscale endpoint |
| `http://localhost:8000/health` | Server status |
| `http://localhost:8000/docs` | Interactive Swagger UI |

Users only need one link. No manual backend URL entry unless they want to point the frontend at a different backend.

---

## Run Anywhere — Public URL

### Cloudflare Tunnel (recommended, free, no account needed)

On the GPU server after starting uvicorn:

```bash
# Install cloudflared (one-time)
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared

# Expose the backend + frontend together
./cloudflared tunnel --url http://localhost:8000
```

It prints a public URL like `https://pixelai-random-words.trycloudflare.com`. Anyone can open that URL and immediately use the app — frontend and API work from the same host.

### ngrok

```bash
ngrok http 8000
```

### SSH tunnel (laptop-only)

```bash
ssh -L 8000:localhost:8000 user@gpu-server
```

---

## Architecture

```
  ┌─────────────────────────────────┐
  │     FastAPI Server (port 8000)  │
  │                                 │
  │  /           →  frontend/*      │  ← static files
  │  /api/*      →  AI routes       │
  │  /health     →  status JSON     │
  │  /docs       →  Swagger UI      │
  └────────────┬────────────────────┘
               │
      ┌────────┴─────────┐
      ▼                  ▼
  BiRefNet            Real-ESRGAN
  MiDaS               OpenCV Haar
  (GPU)               (GPU)
```

### Repo layout

```
AI-Era-Image-Editor/
├── frontend/                    # Static SPA — served by FastAPI
│   ├── index.html
│   ├── style.css
│   └── app.js
└── backend/                     # FastAPI service
    ├── main.py                  # entry point — also mounts frontend
    ├── requirements.txt
    ├── run.sh                   # optional startup script
    ├── routes/                  # HTTP endpoints
    ├── services/                # AI logic
    ├── models/weights/          # gitignored — download separately
    ├── utils/
    └── data/                    # runtime uploads + outputs
```

---

## API Reference

| Endpoint | Method | Body | Output |
|----------|--------|------|--------|
| `/api/classify/` | POST | `file` (image) | JSON: score, label, tags, metrics |
| `/api/remove-bg/` | POST | `file`, `alpha_matting` (bool, opt) | PNG (transparent) |
| `/api/blur-bg/` | POST | `file`, `blur_radius` (int 1–60) | JPEG |
| `/api/upscale/` | POST | `file`, `scale` (2 or 4) | PNG + `X-Upscale-Method` header |
| `/health` | GET | — | CUDA + model status |
| `/docs` | GET | — | Interactive Swagger UI |

Input images are capped at 2048px longest edge (1024 for upscale).

---

## Environment Variables

All optional.

| Variable | Default | Purpose |
|----------|---------|---------|
| `PIXELAI_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `PIXELAI_MAX_DIM` | `2048` | Longest-edge cap on input images |
| `PIXELAI_DATA_DIR` | `data` | Where uploads/outputs get stored |
| `PIXELAI_REMBG_MODEL` | `birefnet-general` | rembg model — see list below |
| `PIXELAI_MIDAS_MODEL` | `MiDaS_small` | Depth model — `MiDaS_small` / `DPT_Hybrid` / `DPT_Large` |
| `PIXELAI_BLUR_USE_REMBG` | `1` | Use rembg mask for bg blur (cleaner than Otsu-on-depth) |
| `U2NET_HOME` | `~/.u2net` | Where rembg caches model weights — **set this if home dir isn't writable** |
| `TORCH_HOME` | `~/.cache/torch` | Where torch.hub caches MiDaS |
| `XDG_CACHE_HOME` | `~/.cache` | Fallback cache location |

### rembg model options
- `u2netp` — tiny, fast
- `u2net` — classic
- `isnet-general-use` — sharper edges than U2Net
- `birefnet-general` — **default, state-of-the-art**
- `birefnet-general-lite` — half the size
- `birefnet-portrait` — portraits only

---

## Troubleshooting

### `Connection refused` in the browser
Backend isn't running. Check:
```bash
curl http://localhost:8000/health
```
Make sure uvicorn is bound to `0.0.0.0`, not `127.0.0.1`:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend loads but API calls fail
The frontend auto-detects same-origin API by default. If you see CORS errors, either:
1. Start uvicorn so it serves both (the default when you use `http://localhost:8000/`)
2. Or manually set the backend URL in the topbar input field

### `rembg: false` in `/health`
Either `rembg[gpu]` isn't installed or `U2NET_HOME` points somewhere unwritable.
```bash
pip install "rembg[gpu]"
export U2NET_HOME=/tmp/u2net-cache
mkdir -p "$U2NET_HOME"
```
**Restart uvicorn in the same shell where you exported the env var.**

### `PermissionError: ... /users/.../`
Your home directory isn't writable (common on university clusters). Export `U2NET_HOME` and `TORCH_HOME` to `/tmp/...` before starting uvicorn.

### `No module named 'torchvision.transforms.functional_tensor'`
basicsr + newer torchvision. Patch:
```bash
sed -i 's|torchvision.transforms.functional_tensor|torchvision.transforms.functional|' \
  venv/lib/python3.12/site-packages/basicsr/data/degradations.py
```

### `libcudnn.so.9: cannot open shared object file`
ONNX Runtime's CUDA provider needs cuDNN 9. Either:
1. `pip install nvidia-cudnn-cu12==9.*` and add lib dir to `LD_LIBRARY_PATH`
2. Run rembg on CPU: `pip uninstall onnxruntime-gpu && pip install onnxruntime`

### `realesrgan.last_error: ModuleNotFoundError`
```bash
pip install basicsr realesrgan
sed -i 's|torchvision.transforms.functional_tensor|torchvision.transforms.functional|' \
  venv/lib/python3.12/site-packages/basicsr/data/degradations.py
```

### Upscale returns a blurry image with no improvement
`X-Upscale-Method: lanczos` means Real-ESRGAN didn't load — check `/health` for `realesrgan.last_error`.

### `cuda_available: false` on a GPU machine
PyTorch is the CPU build. Force reinstall:
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Background removal gives jagged results
BiRefNet failed to load and the server fell back to the naive corner-color cutter. Check `/health` for `rembg: true`.

---

## Committing to the repo

```bash
git clone https://github.com/siragena/AI-Era-Image-Editor.git
cd AI-Era-Image-Editor

# First time on this machine only
git config user.name  "Your Name"
git config user.email "you@example.com"

# Make changes:
git status
git add .
git commit -m "Short description"
git pull --rebase
git push
```

---

## Team

**Group 3 — Spring 2026 Capstone**
Kent State University, Department of Computer Science — CS-69099-001

- Karthik Reddy Mylapurapu
- Rajesh Racha — [@siragena](https://github.com/siragena)
- Solange Iragena
- Vinithanjali Ummadi Reddy

---

## Acknowledgments

- [rembg](https://github.com/danielgatis/rembg) — background removal pipeline
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — segmentation model
- [MiDaS](https://github.com/isl-org/MiDaS) — monocular depth estimation
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — super-resolution
- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework
