# What to do on Viper after pulling these fixes

BiRefNet is a new model — the first request will download the weights (~300MB)
which takes a minute. After that it's cached. Nothing to install manually.

However, `rembg` must be installed with GPU support. You saw this warning in
the logs earlier:

    No onnxruntime backend found.
    Please install rembg with CPU or GPU support

Fix on Viper (inside your venv):

    pip install "rembg[gpu]"

Then restart uvicorn. First `/api/remove-bg/` call will download BiRefNet
weights — watch the logs.

## If BiRefNet weights won't download on Viper

Some clusters block HuggingFace/GitHub. You have two options:

1. Change the env var before starting uvicorn to use a smaller, already-cached
   model:

        PIXELAI_REMBG_MODEL=birefnet-general-lite uvicorn main:app ...

   or:

        PIXELAI_REMBG_MODEL=isnet-general-use uvicorn main:app ...

2. Or download BiRefNet weights manually and put them in
   `~/.u2net/` (which is where rembg caches models).

## Model quality ranking (for background removal)

From best to fastest:

    birefnet-general       best edges, 300MB, slowest
    birefnet-general-lite  ~2x faster, slightly worse edges
    isnet-general-use      fast, decent, already small
    u2net                  classic, roughest edges

## Blur and upscale fixes

No extra install needed. Just pull and restart.

- Blur now has adaptive mask softness + fallback for flat-depth images
- Upscale now returns full-resolution ESRGAN output (was being squashed to
  preview size in the frontend — that's what made it look like nothing happened)
- Upscale input capped at 1024px longest edge so 4x output stays ≤ 4096px
