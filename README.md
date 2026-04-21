# PixelAI — AI Image Editor (Group 3, Spring 2026)

Full-stack AI image editor. FastAPI backend running the heavy ML (rembg / MiDaS / Real-ESRGAN) on the Kent State Viper GPU server, plus a single-page vanilla-JS frontend.

```
AI-Era-Image-Editor/
├── frontend/              Static single-page app (open index.html)
│   ├── index.html
│   ├── style.css
│   └── app.js
└── backend/               FastAPI service
    ├── main.py
    ├── requirements.txt
    ├── routes/            HTTP endpoints
    ├── services/          AI logic
    ├── models/            Model loaders + weights folder
    ├── utils/             Image helpers
    └── data/              uploads + outputs (auto-created)
```

---

## 1 · Run the Backend

### Local (CPU, for development)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate                    # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000` — should return `{"status": "PixelAI API is running", "version": "1.1.0"}`
Visit `http://localhost:8000/docs` for the interactive API docs.
Visit `http://localhost:8000/health` to see what models loaded and whether CUDA is available.

### GPU server (Viper / H100)

```bash
# Check CUDA version first
nvidia-smi

# Install GPU PyTorch (replace cu118 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU rembg
pip install rembg[gpu]

# Real-ESRGAN + basicsr
pip install basicsr realesrgan

# Download Real-ESRGAN weights
mkdir -p backend/models/weights
cd backend/models/weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
cd ../../..

# Start the server, bound to all interfaces so SSH-forwarding / tunnels work
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Environment variables (all optional)

| Var | Default | Purpose |
|---|---|---|
| `PIXELAI_CORS_ORIGINS` | `*` | Comma-separated allowed origins, e.g. `https://mysite.com,http://localhost:5500` |
| `PIXELAI_MAX_DIM` | `2048` | Longest-edge cap on input images before processing |
| `PIXELAI_DATA_DIR` | `data` | Where uploads/outputs get stored |
| `PIXELAI_REMBG_MODEL` | `isnet-general-use` | rembg model: `u2net`, `u2netp`, `u2net_human_seg`, `isnet-general-use`, `isnet-anime`, `birefnet-general` |
| `PIXELAI_MIDAS_MODEL` | `MiDaS_small` | Depth model: `MiDaS_small` (fast), `DPT_Hybrid`, `DPT_Large` (slow, best) |

Example:
```bash
PIXELAI_REMBG_MODEL=birefnet-general PIXELAI_MIDAS_MODEL=DPT_Hybrid uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 2 · Run the Frontend

Open `frontend/index.html` in any browser. Done.

For best results serve it through a simple static server (avoids some CORS quirks):
```bash
cd frontend
python3 -m http.server 5500
# then visit http://localhost:5500
```

### Point the frontend at any backend

The topbar has a **server URL input** with a server icon next to it. Paste any reachable backend URL there and press the checkmark (or Enter). It is saved in `localStorage` so it persists across reloads. Examples:

- `http://localhost:8000` (local)
- `http://localhost:8001` (local on a custom port)
- `https://pixelai-abc123.trycloudflare.com` (Cloudflare Tunnel)
- `https://abcd-12-34-56-78.ngrok-free.app` (ngrok)

The footer of the AI Tools menu shows live server status (`GPU: NVIDIA H100`, `Server online (CPU)`, or `Server: offline`).

---

## 3 · Run Anywhere — expose Viper backend to the public internet

The backend was previously only reachable via SSH port-forward, meaning it only worked on the laptop with the tunnel open. The three options below let you run the backend on Viper and access it from **any browser, anywhere in the world**.

### Option A — Cloudflare Tunnel (recommended, free, no account needed for quick)

On Viper, after starting uvicorn on port 8000:
```bash
# One-time install
# macOS: brew install cloudflared
# Linux: wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared

./cloudflared tunnel --url http://localhost:8000
```
It prints a URL like `https://pixelai-random-words.trycloudflare.com`. Copy that into the frontend's server URL input. Done — it works from any network.

For a stable named URL, create a Cloudflare account, run `cloudflared tunnel login`, and follow the named-tunnel docs.

### Option B — ngrok (free tier, session-limited)

```bash
# One-time: sign up at ngrok.com, grab authtoken
ngrok config add-authtoken YOUR_TOKEN

# Then:
ngrok http 8000
```
Paste the printed `https://...ngrok-free.app` URL into the frontend.

### Option C — SSH tunnel (traditional, laptop-only)

```bash
ssh -L 8000:localhost:8000 rracha@viper.cs.kent.edu
# Keep this terminal open. Frontend points at http://localhost:8000.
```

### Security note
Options A and B expose your backend to the whole internet. Set `PIXELAI_CORS_ORIGINS` to your actual frontend origin(s) in production, and consider adding a lightweight API key header check in `main.py` before demoing publicly.

---

## 4 · Host the frontend too (optional)

Because the frontend is three static files, any static host works:

- **GitHub Pages** — push the `frontend/` folder to a `gh-pages` branch, enable Pages on that branch.
- **Netlify / Vercel** — drag-and-drop `frontend/` into their web UI.
- **Cloudflare Pages** — connect the repo, set build dir to `frontend`.

Once hosted, anyone with the link can use PixelAI — they just paste the backend URL into the server input in the topbar.

---

## 5 · API reference

| Endpoint | Method | Body | Output |
|---|---|---|---|
| `/api/classify/` | POST | `file` (image) | JSON: score, label, tags, metrics |
| `/api/remove-bg/` | POST | `file`, `alpha_matting` (bool, opt) | PNG (transparent bg) |
| `/api/blur-bg/` | POST | `file`, `blur_radius` (int 1–60) | JPEG |
| `/api/upscale/` | POST | `file`, `scale` (2 or 4) | PNG (+ `X-Upscale-Method` header: `realesrgan` or `lanczos`) |
| `/health` | GET | — | CUDA + model status |

All image endpoints accept JPEG, PNG, or WEBP.

---

## 6 · Committing to the repo from your side

```bash
# Clone (first time only)
git clone https://github.com/siragena/AI-Era-Image-Editor.git
cd AI-Era-Image-Editor

# Configure your identity (first time on this machine only)
git config user.name  "Your Name"
git config user.email "you@example.com"

# Make changes, then:
git status                           # see what changed
git add .                            # stage everything, or `git add path/to/file`
git commit -m "Fix crop aspect ratio math and AI blur slider"
git pull --rebase                    # in case a teammate pushed
git push
```

### Branch workflow (if multiple teammates)

```bash
git checkout -b feature/my-change    # create + switch to branch
# ...edit...
git add . && git commit -m "..."
git push -u origin feature/my-change  # first push needs -u

# Then open a Pull Request on GitHub and merge.
```

### First time pushing this refactor

```bash
# From inside the repo root, copy in the new files (or just drop-replace)
# The structure is identical to the original, so drag-and-drop works.

git add backend frontend README.md
git commit -m "Refactor: cleaner services, env config, configurable backend URL, frontend fixes"
git push
```

### If you hit the big "refusing to merge unrelated histories"
```bash
git pull origin main --allow-unrelated-histories
```

---

## 7 · What changed vs the original

**Backend**
- Classifier re-weighted (exposure 2.0, contrast 2.0, sharpness 2.5, noise 1.0, face 0.5 over a 2.0 base). Added EXIF rotation handling for phone photos.
- Background removal defaults to `isnet-general-use` with alpha matting for softer edges. Override via `PIXELAI_REMBG_MODEL`.
- Background blur uses Otsu thresholding (per-image adaptive) + heavy mask feathering. Defaults to `MiDaS_small` for speed.
- Upscale reports `X-Upscale-Method` header so the UI can warn when ESRGAN fell back to LANCZOS.
- Global exception handler, request timing middleware, `/health` exposes GPU info and which models loaded.
- All config via env vars.

**Frontend**
- Backend URL configurable in the topbar, persisted in localStorage.
- Default API base fixed to `http://localhost:8000` (was `8001`, which didn't match the server).
- Crop aspect ratio math fixed for non-square ratios (4:3 and 16:9 now work correctly).
- Adjustments apply over the current "baseline" — so running Remove BG then adjusting brightness no longer wipes the bg-removed result.
- AI blur slider in the right panel now actually controls the blur radius sent to the server.
- Upscale scale picker no longer closes the dropdown when clicked.
- Offline modal shows three clear options (local / SSH / tunnel) and the actual URL that failed.
