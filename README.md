# bleeper-parakeet

Bleeper profanity filter pipeline using **NVIDIA Parakeet TDT** as the transcription backend, running on the official `nvcr.io/nvidia/nemo:24.09` Docker base image.

> **Sister repo:** [`bleep_test`](https://github.com/clawdbot777/bleep_test) — same pipeline with `faster-whisper` as the default backend. Use both to compare accuracy and speed.

## Model Variants

Set in the job JSON under `whisperx_settings.model`:

| Model | ID | Notes |
|---|---|---|
| Parakeet TDT 0.6b v3 | `nvidia/parakeet-tdt-0.6b-v3` | **Default** — faster, lower VRAM |
| Parakeet TDT 1.1b | `nvidia/parakeet-tdt-1.1b` | Larger — slightly higher accuracy |

## Build & Run

```bash
docker compose up -d --build
```

> Uses port **5001** by default to avoid clashing with `bleep_test` on 5000.

## Submit a Job

```bash
curl -s -X POST http://localhost:5001/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "movie.mkv",
    "whisperx_settings": {
      "model": "nvidia/parakeet-tdt-0.6b-v3"
    }
  }'
```

To try the 1.1b variant:
```bash
"model": "nvidia/parakeet-tdt-1.1b"
```

## Why a Separate Repo?

The NeMo base image is ~20GB and includes all CUDA/PyTorch/nemo_toolkit deps — no pip-installing NeMo into a generic CUDA image, no CUDA torch clobbering issues. Clean base, clean deps.

`bleep_test` stays lean with `faster-whisper` (CTranslate2). This repo stays purpose-built for Parakeet.
