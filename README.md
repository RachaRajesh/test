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
- 20-step undo/redo history
- Before/after comparison slider after AI operations
- Drag-and-drop upload, PNG export
- Keyboard shortcuts (V, C, Ctrl+Z, Ctrl+Y, Ctrl+S)
- "Reset All" reverts to the original uploaded image (wipes AI ops + adjustments)

### Deployment
- Backend URL is configurable from the UI (saved in localStorage)
- Run the server on any GPU machine, expose via Cloudflare Tunnel or ngrok, use from anywhere
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

Open `frontend/index.html` in your browser. That's it.

### Option B — GPU Server (full setup with every gotcha we hit)

**This is the setup that actually works on a shared GPU server with a read-only home directory.** Derived from getting this running on Kent State's Viper (NVIDIA H100, CUDA 12.4).

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

# 7. PATCH basicsr — it imports a torchvision function that was moved in newer versions.
#    Without this patch, `from realesrgan import RealESRGANer` will crash.
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

### Startup script (recommended)

To avoid retyping the env vars every time, save a startup script:

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

Next time just run `./run.sh`.

### Verify the server is actually working

```bash
curl http://localhost:8000/health | python3 -m json.tool
```

You want to see:
```json
{
  "status": "ok",
  "cuda_available": true,
  "device": "NVIDIA H100 NVL",
  "torch_version": "2.6.0+cu124",
  "models": {
    "rembg": true,
    "realesrgan": {
      "available": {"x2": true, "x4": true},
      "last_error": null
    }
  }
}
```

Common failures:
- `cuda_available: false` → PyTorch is the CPU build, redo step 3 with `--force-reinstall`
- `rembg: false` → `U2NET_HOME` wasn't set before uvicorn started
- `realesrgan.last_error` is not `null` → basicsr patch (step 7) didn't run or `pip install basicsr realesrgan` failed

---

## Run Anywhere — Public URL

The frontend has a **server URL input in the top bar**. Paste any reachable backend URL and it connects. This lets you demo the app from anywhere.

### Cloudflare Tunnel (recommended, free, no account needed)

On the GPU server after starting uvicorn:

```bash
# Install cloudflared (one-time)
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared

# Expose the backend
./cloudflared tunnel --url http://localhost:8000
```

It prints a public URL like `https://pixelai-random-words.trycloudflare.com`. Paste that into the server URL input in the frontend topbar and click the checkmark.

### ngrok

```bash
ngrok http 8000
```

Paste the printed `https://...ngrok-free.app` URL into the frontend.

### SSH tunnel (laptop-only)

```bash
ssh -L 8000:localhost:8000 user@gpu-server
```

---

## Architecture

```
┌────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│   Browser      │─────▶│   FastAPI Server   │─────▶│   AI Models        │
│   (Vanilla JS) │◀─────│   (Python 3.12)    │◀─────│   (PyTorch / ONNX) │
└────────────────┘      └────────────────────┘      └────────────────────┘
   HTML/CSS/JS          /api/classify                BiRefNet  (rembg)
   Canvas API           /api/remove-bg               MiDaS     (depth)
   localStorage         /api/blur-bg                 Real-ESRGAN (SR)
                        /api/upscale                 OpenCV Haar (faces)
                        /health
```

### Repo layout

```
AI-Era-Image-Editor/
├── frontend/                    # Static SPA — no build step
│   ├── index.html
│   ├── style.css
│   └── app.js
└── backend/                     # FastAPI service
    ├── main.py                  # entry point
    ├── requirements.txt
    ├── run.sh                   # optional startup script
    ├── routes/                  # HTTP endpoints
    ├── services/                # AI logic
    ├── models/weights/          # gitignored — download separately
    ├── utils/
    └── data/                    # runtime uploads + outputs (auto-created)
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

All image endpoints accept JPEG, PNG, or WEBP. Input is resized to 2048 px longest edge before processing (1024 for upscale).

---

## Environment Variables

All optional — sensible defaults for everything.

| Variable | Default | Purpose |
|----------|---------|---------|
| `PIXELAI_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `PIXELAI_MAX_DIM` | `2048` | Longest-edge cap on input images |
| `PIXELAI_DATA_DIR` | `data` | Where uploads/outputs get stored |
| `PIXELAI_REMBG_MODEL` | `birefnet-general` | rembg model — see below |
| `PIXELAI_MIDAS_MODEL` | `MiDaS_small` | Depth model — `MiDaS_small` / `DPT_Hybrid` / `DPT_Large` |
| `PIXELAI_BLUR_USE_REMBG` | `1` | Use rembg mask for bg blur (cleaner than Otsu-on-depth) |
| `U2NET_HOME` | `~/.u2net` | Where rembg caches model weights — **set this if home dir isn't writable** |
| `TORCH_HOME` | `~/.cache/torch` | Where torch.hub caches MiDaS — same reasoning |
| `XDG_CACHE_HOME` | `~/.cache` | Fallback cache location for some libs |

### rembg model options (fastest → highest quality)
- `u2netp` — tiny, fast
- `u2net` — classic
- `isnet-general-use` — sharper edges
- `birefnet-general` — **default, best quality**
- `birefnet-general-lite` — half the size
- `birefnet-portrait` — portraits only

---

## Troubleshooting

### `Connection refused` from the frontend
The backend isn't running or the tunnel is broken. Check:
```bash
curl http://localhost:8000/
# Should return: {"status":"PixelAI API is running","version":"1.1.0"}
```
Make sure uvicorn is bound to `0.0.0.0`, not `127.0.0.1`:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### `rembg: false` in `/health`
Either `rembg[gpu]` isn't installed or `U2NET_HOME` points somewhere unwritable.
```bash
pip install "rembg[gpu]"
export U2NET_HOME=/tmp/u2net-cache
mkdir -p "$U2NET_HOME"
```
**Restart uvicorn in the same shell where you exported the env var** — subprocesses don't inherit shell state from a different terminal.

### `PermissionError: ... /users/kent/student/...`
Your home directory isn't writable (common on university clusters). Export `U2NET_HOME` and `TORCH_HOME` to `/tmp/...` before starting uvicorn.

### `No module named 'torchvision.transforms.functional_tensor'`
Basicsr + newer torchvision incompatibility. Patch:
```bash
sed -i 's|torchvision.transforms.functional_tensor|torchvision.transforms.functional|' \
  venv/lib/python3.12/site-packages/basicsr/data/degradations.py
```

### `libcudnn.so.9: cannot open shared object file`
ONNX Runtime's CUDA provider needs cuDNN 9. Either:
1. `pip install nvidia-cudnn-cu12==9.*` and add the lib dir to `LD_LIBRARY_PATH`
2. Run rembg on CPU: `pip uninstall onnxruntime-gpu && pip install onnxruntime`

### `realesrgan.last_error: ModuleNotFoundError`
```bash
pip install basicsr realesrgan
sed -i 's|torchvision.transforms.functional_tensor|torchvision.transforms.functional|' \
  venv/lib/python3.12/site-packages/basicsr/data/degradations.py
python -c "from realesrgan import RealESRGANer; print('OK')"
```

### Upscale returns a blurry image with no visible improvement
`X-Upscale-Method: lanczos` means Real-ESRGAN didn't load — check `/health` for `realesrgan.last_error`.

### `cuda_available: false` on a GPU machine
PyTorch is the CPU build. Force reinstall:
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Should print: 2.6.0+cu124 True
```

### Background removal gives jagged, hole-ridden results
BiRefNet failed to load and the server fell back to the naive corner-color cutter. Check `/health` for `rembg: true`. Most common cause: `U2NET_HOME` not set.

---

## Committing to the repo

```bash
# Clone (first time only)
git clone https://github.com/siragena/AI-Era-Image-Editor.git
cd AI-Era-Image-Editor

# First time on this machine only
git config user.name  "Your Name"
git config user.email "you@example.com"

# Make changes, then:
git status
git add .
git commit -m "Short description of change"
git pull --rebase
git push
```

### Branch workflow

```bash
git checkout -b feature/my-change
# ...edit...
git add . && git commit -m "Add my feature"
git push -u origin feature/my-change
# Then open a PR on GitHub
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
