# ─────────────────────────────────────────────────────────────────────────────
# bleeper-parakeet – Automated Profanity Filter Pipeline
# Base: NVIDIA NeMo 24.09 (official image — includes CUDA, PyTorch, nemo_toolkit[asr])
#
# Build:
#   docker build -t bleeper-parakeet .
#
# Run:
#   docker run --gpus all -p 5000:5000 \
#     -v /mnt/user/media:/media \
#     -v /mnt/user/appdata/bleeper-parakeet/uploads:/app/uploads \
#     bleeper-parakeet
#
# Model variants (set in job JSON as "model"):
#   nvidia/parakeet-tdt-0.6b-v3   (default — faster, lower VRAM)
#   nvidia/parakeet-tdt-1.1b       (larger — slightly higher accuracy)
# ─────────────────────────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/nemo:24.09

# NeMo 24.09 base image already includes:
#   - Python 3.10
#   - CUDA 12.x + cuDNN
#   - PyTorch (CUDA build)
#   - torchaudio
#   - nemo_toolkit[asr]
#   - ffmpeg

ENV DEBIAN_FRONTEND=noninteractive

# ── Extra system deps not in the NeMo base ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (NeMo/torch/CUDA already present in base) ─────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .
RUN mkdir -p /app/uploads /media

ENV BLEEPER_UPLOAD=/app/uploads \
    WATCH_FOLDER=/media/incoming \
    PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "-m", "gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "600", "run:app"]
