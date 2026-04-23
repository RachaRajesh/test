# PixelAI — AI-Powered Image Editor

> Capstone Project · Spring 2026 · Kent State University · Group 3

A full-stack image editor that combines classical editing tools with modern AI models — background removal, depth-aware blur, AI upscaling, and photo quality scoring — all accessible from a browser, powered by a FastAPI backend running on an NVIDIA H100 GPU.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+cu124-EE4C2C)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

### AI Tools
| Tool | Model | Notes |
|------|-------|-------|
| **Background Removal** | BiRefNet (via rembg) | Alpha-matted output for clean hair/fur edges. PNG with transparency. |
| **Depth-Aware Blur** | BiRefNet mask + MiDaS | Portrait-mode style background blur. Adjustable radius 1–60. |
| **AI Upscaling** | Real-ESRGAN x2/x4 | Super-resolution for low-res photos. Falls back to LANCZOS if weights missing. |
| **Photo Quality Classifier** | Classical CV metrics | 1–10 score using sharpness, exposure, contrast, noise, and face detection. |

### Classical Editing
- Brightness, contrast, saturation, blur sliders (live, non-destructive)
- Crop with aspect-ratio lock (free / 1:1 / 4:3 / 16:9)
- 20-step undo/redo history
- Before/after comparison slider after AI operations
- Drag-and-drop file upload, PNG export
- Keyboard shortcuts (V, C, Ctrl+Z, Ctrl+Y, Ctrl+S)

### Deployment
- Backend URL is configurable from the UI (saved in localStorage)
- Run the server on any GPU machine, expose via Cloudflare Tunnel or ngrok, use from anywhere in the world
- Lightweight frontend (three static files — no build step)

---

## Quick Start

### Option A: Run locally (CPU)

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

### Option B: GPU server (recommended for real demos)

```bash
# After cloning and entering backend/
python3 -m venv venv
source venv/bin/activate

# GPU PyTorch — check your CUDA version with `nvidia-smi` first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "rembg[gpu]"
pip install basicsr realesrgan

# Download Real-ESRGAN weights
mkdir -p models/weights
cd models/weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
cd ../..

# rembg models auto-download to $U2NET_HOME on first use
export U2NET_HOME=/tmp/u2net-cache
export TORCH_HOME=/tmp/torch-cache
mkdir -p "$U2NET_HOME" "$TORCH_HOME"

uvicorn main:app --host 0.0.0.0 --port 8000
```

See `/health` endpoint to confirm GPU and model availability:

```bash
curl http://localhost:8000/health
# {"status":"ok","cuda_available":true,"device":"NVIDIA H100 NVL",
#  "models":{"rembg":true,"realesrgan":{"available":{"x2":true,"x4":true}}}}
```

---

## Run Anywhere — Public URL

The frontend has a **server URL input in the top bar**. Paste any reachable backend URL and it connects. This lets you demo the app from any machine in the world.

### Cloudflare Tunnel (recommended, free, no account needed)

On the GPU server after starting uvicorn:

```bash
# Install cloudflared (one-time)
# Linux:
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared

# Expose the backend
./cloudflared tunnel --url http://localhost:8000
```

It prints a public URL like `https://pixelai-random-words.trycloudflare.com`. Paste that into the server URL field in the frontend. Done.

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
┌────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│                │      │                    │      │                    │
│   Browser      │─────▶│   FastAPI Server   │─────▶│   AI Models        │
│   (Vanilla JS) │◀─────│   (Python 3.12)    │◀─────│   (PyTorch / ONNX) │
│                │      │                    │      │                    │
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
    ├── routes/                  # HTTP endpoints
    │   ├── classifier.py
    │   ├── bgremove.py
    │   ├── bgblur.py
    │   └── upscale.py
    ├── services/                # AI logic
    │   ├── classifier.py        # classical CV quality scorer
    │   ├── bgremove.py          # rembg + BiRefNet
    │   ├── bgblur.py            # BiRefNet mask + Gaussian blur
    │   └── upscale.py           # Real-ESRGAN
    ├── models/                  # Model loaders + weights folder
    │   └── weights/             # (gitignored — download separately)
    ├── utils/
    │   └── image_io.py          # upload/output helpers
    └── data/                    # runtime uploads + outputs (auto-created)
```

---

## API Reference

| Endpoint | Method | Body | Output |
|----------|--------|------|--------|
| `/api/classify/` | POST | `file` (image) | JSON: score, label, tags, metrics |
| `/api/remove-bg/` | POST | `file`, `alpha_matting` (bool, optional) | PNG (transparent) |
| `/api/blur-bg/` | POST | `file`, `blur_radius` (int 1–60) | JPEG |
| `/api/upscale/` | POST | `file`, `scale` (2 or 4) | PNG + `X-Upscale-Method` header |
| `/health` | GET | — | CUDA + model status |
| `/docs` | GET | — | Interactive Swagger UI |

All image endpoints accept JPEG, PNG, or WEBP. Input is resized to 2048 px longest edge before processing (1024 for upscale — output would be too huge otherwise).

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
If that fails on the server itself, uvicorn isn't running. If it works on the server but not your laptop, the tunnel is broken.

### `rembg: false` in /health
Either `rembg[gpu]` isn't installed or `U2NET_HOME` points somewhere unwritable.
```bash
pip install "rembg[gpu]"
export U2NET_HOME=/tmp/u2net-cache  # or any writable path
mkdir -p "$U2NET_HOME"
```
Restart uvicorn **in the same shell** where you exported the env var.

### `PermissionError: ... /users/kent/student/...`
Your home directory isn't writable (common on university clusters). Export `U2NET_HOME` and `TORCH_HOME` to a writable path (like `/tmp/project/...`) before starting uvicorn.

### `libcudnn.so.9: cannot open shared object file`
ONNX Runtime's CUDA provider needs cuDNN 9. Either:
1. Install it: `pip install nvidia-cudnn-cu12==9.*` and add to `LD_LIBRARY_PATH`
2. Run rembg on CPU instead: `pip uninstall onnxruntime-gpu && pip install onnxruntime` (BiRefNet on CPU is ~10–30s per image)

### Upscale returns a blurry image
The response header `X-Upscale-Method` will say `lanczos` when Real-ESRGAN isn't loaded — check the weights are in `backend/models/weights/`. LANCZOS is a classical algorithm that just resizes pixels; it has no AI component.

### `cuda_available: false` but I'm on a GPU machine
PyTorch is the CPU build. Force reinstall the GPU wheel:
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Screenshots

*(Add screenshots here after a successful demo — drag images into the repo and link)*

---

## Contributing

PRs welcome from team members. Standard flow:

```bash
git checkout -b feature/my-change
# ...edit, test...
git add . && git commit -m "Add my feature"
git push -u origin feature/my-change
```

Then open a Pull Request on GitHub.

---

## Team

**Group 3 — Spring 2026 Capstone**  
Kent State University, Department of Computer Science

- [Sir Agena](https://github.com/siragena)
- *(add teammates here)*

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [rembg](https://github.com/danielgatis/rembg) — background removal
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — segmentation model
- [MiDaS](https://github.com/isl-org/MiDaS) — monocular depth
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — super-resolution
- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework
