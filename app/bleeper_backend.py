"""
bleeper_backend.py - Core processing logic for the Bleeper automation pipeline.

Pipeline stages:
  1. analyze_and_select_audio_stream  - probe & pick best audio stream
  2. normalize_audio_stream           - transcode any codec → AC3 5.1 (optional, configurable)
  3. extract_audio_stream             - extract selected stream + center (FC) channel
  4. transcribe_audio                 - faster-whisper STT → JSON + SRT
  5. redact_audio                     - mute/bleep profanity, redact subtitles
  6. combine_media_file               - rebuild MKV with family + original audio tracks
  7. cleanup_job_files                - remove temp files, rename output to original name

API routes (Flask):
  POST /api/initialize_job
  POST /api/select_remote_file
  POST /api/upload
  POST /api/analyze_and_select_audio
  POST /api/normalize_audio          (optional pre-processing step)
  POST /api/extract_audio
  POST /api/transcribe
  POST /api/redact
  POST /api/combine_media
  POST /api/cleanup
  POST /api/process_full             (chains all steps end-to-end, fire-and-forget)
  GET  /api/list_files
  GET  /api/job_status/<job_id>
"""

import os
import sys
import logging
import json
import glob
import subprocess
import uuid
import threading
import traceback
import shutil
import tempfile
import string
import re
import time
import magic

from flask import request, jsonify, send_file
from app import app
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Section 1 – Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Job state tracker  {job_id: 'queued'|'running'|'completed'|'failed'}
_job_status_lock = threading.Lock()
job_status: dict[str, str] = {}


def _set_job_status(job_id: str, status: str) -> None:
    with _job_status_lock:
        job_status[job_id] = status


def _get_job_status(job_id: str) -> str:
    with _job_status_lock:
        return job_status.get(job_id, "unknown")


# ---------------------------------------------------------------------------
# Single-worker pipeline queue — ensures jobs run one at a time so the GPU
# isn't hammered by concurrent whisperX/ffmpeg processes.
# ---------------------------------------------------------------------------
import queue as _queue

_pipeline_queue: _queue.Queue = _queue.Queue()
_pipeline_queue_order: list = []   # track position for status reporting
_pipeline_queue_lock = threading.Lock()


def _pipeline_worker() -> None:
    """Single background worker that drains the pipeline queue."""
    while True:
        job_id, fn, args = _pipeline_queue.get()
        with _pipeline_queue_lock:
            if job_id in _pipeline_queue_order:
                _pipeline_queue_order.remove(job_id)
        try:
            fn(*args)
        except Exception as e:
            logger.error(f"[queue] Unhandled error for job {job_id}: {e}")
        finally:
            _pipeline_queue.task_done()


def _queue_pipeline(job_id: str, fn, args: tuple) -> int:
    """Enqueue a pipeline job. Returns queue position (1 = next up)."""
    with _pipeline_queue_lock:
        _pipeline_queue_order.append(job_id)
        position = len(_pipeline_queue_order)
    _pipeline_queue.put((job_id, fn, args))
    return position


# Start the single worker thread (daemon so it dies with the process)
_worker_thread = threading.Thread(target=_pipeline_worker, daemon=True)
_worker_thread.start()
logger.info("Pipeline queue worker started.")

UPLOAD_FOLDER    = os.environ.get("BLEEPER_UPLOAD", "/app/uploads")
TEMP_FOLDER      = "/tmp/uploads/tmp"
PROCESSED_FOLDER = "/tmp/uploads/processed"
FILTER_LIST_PATH = os.path.join(os.path.dirname(__file__), "filter_list.txt")

ALLOWED_EXTENSIONS   = {"mp3", "wav", "ogg", "mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH   = 80 * 1024 * 1024 * 1024  # 80 GB
CHUNK_SIZE           = 20 * 1024 * 1024           # 20 MB

# Normalization target: everything gets converted to this before bleeping.
# Set NORMALIZE_TARGET_CODEC = None to skip normalization.
NORMALIZE_TARGET_CODEC    = "ac3"
NORMALIZE_TARGET_CHANNELS = 6          # 5.1
NORMALIZE_TARGET_BITRATE  = "640k"
NORMALIZE_TARGET_LAYOUT   = "5.1"

for folder in [UPLOAD_FOLDER, TEMP_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)
tempfile.tempdir = TEMP_FOLDER

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["CHUNK_SIZE"]         = CHUNK_SIZE

# Try CUDA; fall back gracefully
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available – using CPU")
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("torch not installed – CUDA unavailable")

# ---------------------------------------------------------------------------
# Channel layout map
# ---------------------------------------------------------------------------

CHANNEL_LAYOUTS: dict[str, list[str]] = {
    "mono":             ["FC"],
    "stereo":           ["FL", "FR"],
    "2.1":              ["FL", "FR", "LFE"],
    "3.0":              ["FL", "FR", "FC"],
    "3.0(back)":        ["FL", "FR", "BC"],
    "4.0":              ["FL", "FR", "FC", "BC"],
    "quad":             ["FL", "FR", "BL", "BR"],
    "quad(side)":       ["FL", "FR", "SL", "SR"],
    "3.1":              ["FL", "FR", "FC", "LFE"],
    "5.0":              ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)":        ["FL", "FR", "FC", "SL", "SR"],
    "4.1":              ["FL", "FR", "FC", "LFE", "BC"],
    "5.1":              ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)":        ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.0":              ["FL", "FR", "FC", "BC", "SL", "SR"],
    "6.0(front)":       ["FL", "FR", "FLC", "FRC", "SL", "SR"],
    "hexagonal":        ["FL", "FR", "FC", "BL", "BR", "BC"],
    "6.1":              ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "6.1(back)":        ["FL", "FR", "FC", "LFE", "BL", "BR", "BC"],
    "6.1(front)":       ["FL", "FR", "LFE", "FLC", "FRC", "SL", "SR"],
    "7.0":              ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
    "7.0(front)":       ["FL", "FR", "FC", "FLC", "FRC", "SL", "SR"],
    "7.1":              ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)":        ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"],
    "7.1(wide-side)":   ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
    "octagonal":        ["FL", "FR", "FC", "BL", "BR", "BC", "SL", "SR"],
}

# Layouts that carry a dedicated center channel
FC_CAPABLE_LAYOUTS = {k for k, v in CHANNEL_LAYOUTS.items() if "FC" in v}

# ---------------------------------------------------------------------------
# Bleep tone configuration — tune these to taste
# ---------------------------------------------------------------------------

BLEEP_FREQUENCY     = 1000      # Hz — 1 kHz is the standard broadcast bleep tone
BLEEP_VOLUME        = 0.5       # 0.0–1.0 — relative to center-channel level
BLEEP_FADE_DURATION = 0.025     # seconds — smooth ramp in/out at bleep edges (25 ms)

# ---------------------------------------------------------------------------
# Section 2 – Helper utilities
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _hwaccel_flags(video: bool = True) -> list[str]:
    """Return ffmpeg hardware-acceleration flags when CUDA is available.

    Only use for commands that process video.  Audio-only commands should
    pass ``video=False`` (or simply omit hwaccel flags).
    """
    if CUDA_AVAILABLE and video:
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    return []


def _run(cmd: list[str], step: str = "") -> None:
    """Run a subprocess command; raise RuntimeError with output on failure."""
    label = f"[{step}] " if step else ""
    logger.debug(f"{label}Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        combined = (result.stdout + "\n" + result.stderr).strip()
        logger.error(f"{label}Command failed:\n{combined}")
        raise RuntimeError(f"{step} failed: {combined[-5000:]}")
    return result


def _probe(path: str) -> dict:
    """Return ffprobe JSON for *path*."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Section 3 – Job config persistence
# ---------------------------------------------------------------------------

def find_job_by_filename(filename: str) -> dict | None:
    """
    Look for an existing incomplete job for the given filename.
    Returns the config dict if found and not yet completed, else None.
    """
    basename = os.path.basename(filename)
    for config_file in glob.glob(os.path.join(UPLOAD_FOLDER, "*_config.json")):
        try:
            with open(config_file) as f:
                config = json.load(f)
            if (config.get("original_filename") == basename or
                    config.get("input_filepath") == filename):
                status = config.get("pipeline_status", "")
                if status not in ("completed",):
                    return config
        except (json.JSONDecodeError, OSError):
            continue
    return None


def get_config(job_id: str) -> dict | None:
    config_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_config.json")
    if not os.path.exists(config_path):
        logger.error(f"Config not found for job_id: {job_id}")
        return None
    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for job_id {job_id}: {e}")
        return None


def update_config(job_id: str, updates: dict) -> None:
    config = get_config(job_id) or {"job_id": job_id}
    config.update(updates)
    config_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Section 4 – Stage 1: Analyze & select best audio stream
# ---------------------------------------------------------------------------

def _stream_priority(stream: dict) -> float:
    """Score an audio stream; higher = better candidate for bleeping."""
    codec    = stream.get("codec_name", "")
    language = stream.get("tags", {}).get("language", "").lower()
    channels = int(stream.get("channels", 0))
    score    = 0.0

    # Language
    if language == "eng":
        score += 1000

    # Codec preference: ac3 multichannel > dts > truehd > eac3 > stereo ac3
    codec_scores = {"ac3": 500, "dts": 400, "truehd": 300, "eac3": 200}
    score += codec_scores.get(codec, 50)
    if codec == "ac3" and channels <= 2:
        score -= 400           # Stereo AC3 is not ideal for FC extraction

    # Prefer multichannel (FC extraction needs ≥3 channels)
    if channels >= 6:
        score += 100
    elif channels >= 3:
        score += 50

    # Bit rate
    try:
        score += float(stream.get("bit_rate", 0)) / 1_000_000
    except (ValueError, TypeError):
        pass

    # Sample rate
    try:
        score += float(stream.get("sample_rate", 0)) / 1000
    except (ValueError, TypeError):
        pass

    logger.debug(f"Stream {stream.get('index')} ({codec} {channels}ch {language}) score={score:.1f}")
    return score


def analyze_and_select_audio_stream(job_id: str) -> dict:
    """Probe the input file and choose the best audio stream."""
    config = get_config(job_id)
    input_file = config.get("input_filename")
    if not input_file:
        raise ValueError(f"No input_filename in config for job_id: {job_id}")

    # Use stored absolute path if available (e.g. passed directly from arr hook)
    # otherwise fall back to uploads folder
    input_path = config.get("input_filepath") or os.path.join(UPLOAD_FOLDER, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    info = _probe(input_path)
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]

    if not audio_streams:
        raise ValueError("No audio streams found in input file.")

    selected = max(audio_streams, key=_stream_priority)
    index_nr = audio_streams.index(selected)

    update_config(job_id, {
        "audio_stream_index_nr": index_nr,
        "selected_stream_info": selected,
    })
    logger.info(f"Selected audio stream #{index_nr}: {selected.get('codec_name')} "
                f"{selected.get('channels')}ch")
    return selected


# ---------------------------------------------------------------------------
# Section 5 – Stage 2: Normalize audio (optional)
# Converts any codec/layout → NORMALIZE_TARGET_CODEC (default: AC3 5.1)
# This eliminates all downstream codec edge-cases in one shot.
# ---------------------------------------------------------------------------

def normalize_audio_stream(job_id: str, force: bool = False) -> str | None:
    """
    Transcode the selected audio stream to a normalized format (AC3 5.1 by default).

    Skip when:
      - NORMALIZE_TARGET_CODEC is None, OR
      - The stream is already the target codec + channel count AND force=False.

    Returns the normalized filename, or None if skipped.
    """
    if NORMALIZE_TARGET_CODEC is None:
        logger.info("Normalization disabled (NORMALIZE_TARGET_CODEC=None) – skipping.")
        return None

    config     = get_config(job_id)
    input_file = config.get("input_filename")
    index_nr   = config.get("audio_stream_index_nr")
    stream     = config.get("selected_stream_info", {})

    src_codec    = stream.get("codec_name", "")
    src_channels = int(stream.get("channels", 0))

    already_normalized = (
        src_codec == NORMALIZE_TARGET_CODEC
        and src_channels == NORMALIZE_TARGET_CHANNELS
    )
    if already_normalized and not force:
        logger.info(
            f"Stream already {NORMALIZE_TARGET_CODEC} {NORMALIZE_TARGET_CHANNELS}ch – "
            "skipping normalization."
        )
        update_config(job_id, {"normalization_skipped": True})
        return None

    input_path   = config.get("input_filepath") or os.path.join(UPLOAD_FOLDER, input_file)
    base          = os.path.splitext(input_file)[0]
    norm_file     = f"{base}_norm.{NORMALIZE_TARGET_CODEC}"
    norm_path     = os.path.join(UPLOAD_FOLDER, norm_file)

    cmd = (
        ["ffmpeg", "-y"]
        + _hwaccel_flags(video=False)
        + [
            "-i", input_path,
            "-map", f"0:a:{index_nr}",
            "-c:a", NORMALIZE_TARGET_CODEC,
            "-b:a", NORMALIZE_TARGET_BITRATE,
            "-ac", str(NORMALIZE_TARGET_CHANNELS),
            "-ar", "48000",
            "-strict", "-2",
            norm_path,
        ]
    )
    _run(cmd, step="normalize_audio")

    update_config(job_id, {
        "normalized_audio_file": norm_file,
        "normalization_skipped": False,
        "normalized_codec":      NORMALIZE_TARGET_CODEC,
        "normalized_channels":   NORMALIZE_TARGET_CHANNELS,
        "normalized_layout":     NORMALIZE_TARGET_LAYOUT,
    })
    logger.info(f"Normalized audio → {norm_file}")
    return norm_file


# ---------------------------------------------------------------------------
# Section 6 – Stage 3: Extract audio stream + center channel
# ---------------------------------------------------------------------------

def _extract_center_channel(input_path: str, codec: str, bit_rate: int,
                             sample_rate: int, base: str) -> tuple[str, str]:
    """
    Extract the center (FC) channel as a mono audio file.
    Returns (center_filename, center_path).
    """
    if codec == "dts":
        center_file = f"{base}_center.wav"
        center_codec_args = ["-c:a", "pcm_s32le"]
    else:
        center_file = f"{base}_center.{codec}"
        center_codec_args = []

    center_path = os.path.join(UPLOAD_FOLDER, center_file)

    cmd = (
        ["ffmpeg", "-y"]
        + _hwaccel_flags(video=False)
        + ["-i", input_path,
           "-filter_complex", "[0:a]pan=mono|c0=FC[center]",
           "-map", "[center]"]
        + center_codec_args
    )
    if bit_rate:
        cmd += ["-b:a", str(bit_rate)]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-strict", "-2", center_path]

    _run(cmd, step="extract_center_channel")
    return center_file, center_path


def _extract_flr_channel(input_path: str, codec: str, bit_rate: int,
                          sample_rate: int, base: str) -> tuple[str, str]:
    """
    Extract FL+FR as a stereo audio file for independent redaction.
    Returns (flr_filename, flr_path).
    """
    if codec == "dts":
        flr_file        = f"{base}_flr.wav"
        flr_codec_args  = ["-c:a", "pcm_s32le"]
    else:
        flr_file        = f"{base}_flr.{codec}"
        flr_codec_args  = []

    flr_path = os.path.join(UPLOAD_FOLDER, flr_file)

    cmd = (
        ["ffmpeg", "-y"]
        + _hwaccel_flags(video=False)
        + ["-i", input_path,
           "-filter_complex", "[0:a]pan=stereo|c0=FL|c1=FR[flr]",
           "-map", "[flr]"]
        + flr_codec_args
    )
    if bit_rate:
        cmd += ["-b:a", str(bit_rate)]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-strict", "-2", flr_path]

    _run(cmd, step="extract_flr_channel")
    return flr_file, flr_path


def _extract_loudnorm_json(output: str) -> dict | None:
    """
    Robustly extract the loudnorm JSON block from ffmpeg stderr.
    Filenames may contain { } (e.g. {tmdb-1214931}) so we can't just
    use find('{') — scan all blocks for one containing 'input_i'.
    """
    import re
    for match in re.finditer(r'\{[^{}]+\}', output, re.DOTALL):
        try:
            candidate = json.loads(match.group())
            if "input_i" in candidate:
                return candidate
        except json.JSONDecodeError:
            continue
    return None


def _measure_loudness(audio_path: str) -> float:
    """Return integrated loudness (input_i) of an audio file using loudnorm."""
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-filter:a", "loudnorm=print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr or result.stdout

    info = _extract_loudnorm_json(output)
    if not info:
        raise ValueError("No JSON from loudnorm – cannot measure loudness.")
    # Remove the old fallback regex block that was here

    input_i = info.get("input_i")
    if input_i is None:
        raise ValueError("input_i missing from loudnorm JSON.")
    return float(input_i)


def extract_audio_stream(job_id: str) -> dict:
    """
    Extract the best audio stream (post-normalization if applicable) and
    isolate the center channel.  Updates config with all relevant paths.
    """
    config   = get_config(job_id)
    index_nr = config.get("audio_stream_index_nr")

    # Use normalized file if available, otherwise use original input
    norm_file = config.get("normalized_audio_file")
    if norm_file and not config.get("normalization_skipped"):
        source_file   = norm_file
        stream_index  = 0         # normalized file has only one audio stream
        source_codec  = config.get("normalized_codec", NORMALIZE_TARGET_CODEC)
        source_layout = config.get("normalized_layout", NORMALIZE_TARGET_LAYOUT)
        source_path   = os.path.join(UPLOAD_FOLDER, source_file)
    else:
        source_file   = config.get("input_filename")
        stream_index  = index_nr
        # Respect input_filepath for files on mounted media volumes (e.g. Plex)
        # — same pattern as analyze_audio_stream and combine_media_file.
        source_path   = config.get("input_filepath") or os.path.join(UPLOAD_FOLDER, source_file)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Probe the source to get current stream properties
    info = _probe(source_path)
    audio_streams = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise ValueError(f"No audio streams in {source_file}")

    # For a normalized file there is exactly one stream
    stream = audio_streams[min(stream_index, len(audio_streams) - 1)]
    codec        = stream.get("codec_name", "ac3")
    bit_rate     = int(stream.get("bit_rate", 0)) or None
    sample_rate  = int(stream.get("sample_rate", 48000))
    channels     = int(stream.get("channels", 6))
    layout       = stream.get("channel_layout", "5.1")

    base = os.path.splitext(os.path.basename(source_file))[0]

    # ---- Step A: Handle TrueHD by converting to DTS first ----------------
    if codec == "truehd":
        logger.info("TrueHD detected – converting to DTS before extraction.")
        dts_file = f"{base}_audio.dts"
        dts_path = os.path.join(UPLOAD_FOLDER, dts_file)
        cmd = (
            ["ffmpeg", "-y"]
            + _hwaccel_flags(video=False)
            + ["-i", source_path,
               "-map", f"0:a:{stream_index}",
               "-c:a", "dca",
               "-b:a", str(bit_rate) if bit_rate else "1509k",
               "-ac", "6",
               "-strict", "-2",
               dts_path]
        )
        _run(cmd, step="truehd_to_dts")
        source_path  = dts_path
        stream_index = 0
        codec        = "dts"
        bit_rate     = None
        sample_rate  = 48000
        channels     = 6
        layout       = "5.1"
        update_config(job_id, {"audio_filename": dts_file})

    # ---- Step B: Extract full audio stream (copy) -------------------------
    audio_file = f"{base}_audio.{codec if codec != 'dts' else 'dts'}"
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file)
    cmd = (
        ["ffmpeg", "-y"]
        + _hwaccel_flags(video=False)
        + ["-i", source_path,
           "-map", f"0:a:{stream_index}",
           "-c:a", "copy",
           "-strict", "-2",
           audio_path]
    )
    _run(cmd, step="extract_audio_stream")

    # ---- Step C: Re-probe extracted stream for accurate metadata ----------
    info2   = _probe(audio_path)
    stream2 = (info2.get("streams") or [stream])[0]
    codec        = stream2.get("codec_name", codec)
    bit_rate     = int(stream2.get("bit_rate", 0)) or bit_rate
    sample_rate  = int(stream2.get("sample_rate", sample_rate))
    channels     = int(stream2.get("channels", channels))
    layout       = stream2.get("channel_layout", layout)

    # ---- Step D: Extract center channel -----------------------------------
    has_fc = layout in FC_CAPABLE_LAYOUTS or channels >= 3
    if has_fc:
        center_file, center_path = _extract_center_channel(
            audio_path, codec, bit_rate, sample_rate, base
        )
        loudness = _measure_loudness(center_path)
    else:
        # Mono / stereo → use the whole stream as the "center"
        logger.info(f"Layout {layout!r} has no dedicated FC – using full stream as center.")
        center_file  = audio_file
        center_path  = audio_path
        loudness     = _measure_loudness(center_path)

    # ---- Step E: Extract FL+FR stereo (for surround redaction) ------------
    flr_file = None
    if has_fc and "FL" in CHANNEL_LAYOUTS.get(layout, []) and "FR" in CHANNEL_LAYOUTS.get(layout, []):
        try:
            flr_file, _ = _extract_flr_channel(
                audio_path, codec, bit_rate, sample_rate, base
            )
            logger.info(f"FL+FR stereo extracted → {flr_file}")
        except Exception as exc:
            logger.warning(f"FL+FR extraction failed (non-fatal): {exc}")
            flr_file = None

    update_config(job_id, {
        "audio_filename":      audio_file,
        "center_channel_file": center_file,
        "flr_channel_file":    flr_file,
        "loudness_info":       loudness,
        "audio_stream_info":   {
            "streams": [stream2],
            "format":  info2.get("format", {}),
        },
        "source_layout":      layout,
        "source_channels":    channels,
        "source_codec":       codec,
        "source_bit_rate":    bit_rate,
        "source_sample_rate": sample_rate,
    })
    logger.info(f"Center channel extracted → {center_file}  (loudness: {loudness:.1f} LUFS)")
    return {"center_channel_file": center_file, "loudness": loudness}


# ---------------------------------------------------------------------------
# Section 7 – Stage 4: Transcribe center channel (whisperX)
# ---------------------------------------------------------------------------

def transcribe_audio(job_id: str, whisperx_settings: dict | None = None) -> bool:
    config      = get_config(job_id)
    input_file  = config.get("center_channel_file")
    if not input_file:
        raise ValueError("center_channel_file not set. Run extract_audio first.")

    input_path = os.path.join(UPLOAD_FOLDER, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Center channel file not found: {input_path}")

    ws = whisperx_settings or {}
    _transcribe_parakeet(job_id, input_path, input_file, ws)

    # ---- FL+FR pass -------------------------------------------------------
    flr_file = config.get("flr_channel_file")
    if flr_file:
        flr_path = os.path.join(UPLOAD_FOLDER, flr_file)
        if os.path.exists(flr_path):
            logger.info("Transcribing FL+FR channel (VAD requested but not yet supported by Parakeet — transcribing full stream)…")
            _transcribe_parakeet(
                job_id, flr_path, flr_file, ws,
                vad_filter=True,
                config_keys=("flr_transcription_json", "flr_transcription_srt"),
            )
        else:
            logger.warning(f"FL+FR file not found, skipping FL+FR transcription: {flr_path}")

    return True


def _transcribe_parakeet(job_id: str, input_path: str, input_file: str,
                          ws: dict,
                          vad_filter: bool = False,
                          config_keys: tuple[str, str] = ("transcription_json",
                                                           "transcription_srt")) -> None:
    """Transcribe using NVIDIA Parakeet TDT via parakeet_transcribe.py.

    vad_filter — requests VAD pre-filtering; Parakeet doesn't natively support
                 this yet so a warning is logged and the full stream is processed.
    config_keys — (json_key, srt_key) under which output paths are stored.
    """
    model  = ws.get("model", "nvidia/parakeet-tdt-0.6b-v3")
    script = os.path.join(os.path.dirname(__file__), "parakeet_transcribe.py")

    cmd = [
        sys.executable, script,
        input_path,
        "--output_dir", UPLOAD_FOLDER,
        "--model",      model,
    ]
    if vad_filter:
        cmd += ["--vad_filter"]   # accepted by script; logs a warning, no-op for now

    step = "flr_transcribe" if config_keys[0] != "transcription_json" else "parakeet_transcribe"
    _run(cmd, step=step)

    base      = os.path.splitext(os.path.basename(input_file))[0]
    json_file = f"{base}.json"
    srt_file  = f"{base}.srt"
    json_key, srt_key = config_keys
    update_config(job_id, {
        json_key:                json_file,
        srt_key:                 srt_file,
        "transcription_backend": "parakeet",
    })
    logger.info(f"Parakeet transcription complete [{json_key}] → {json_file}, {srt_file}")


# ---------------------------------------------------------------------------
# Section 8 – Stage 5: Redact profanity
# ---------------------------------------------------------------------------

def _load_profanity_list() -> list[str]:
    if not os.path.exists(FILTER_LIST_PATH):
        raise FileNotFoundError(f"Filter list not found: {FILTER_LIST_PATH}")
    with open(FILTER_LIST_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def identify_profanity_timestamps(timestamps_data: dict, profanity: list[str],
                                   pad_before: float = 0.10,
                                   pad_after:  float = 0.10) -> list[dict]:
    """Return list of {start, end} dicts for each profane word."""
    # Clean profanity list the same way audio words are cleaned: strip ALL
    # punctuation (not just edges) so entries like "Christ's" or "face-fucks"
    # match the de-punctuated form whisperX produces ("christs", "facefucks").
    clean_profanity = {
        "".join(c for c in w.lower() if c not in string.punctuation).strip()
        for w in profanity
    }
    hits = []
    for segment in timestamps_data.get("segments", []):
        words = segment.get("words", [])
        for i, word in enumerate(words):
            raw = word.get("word", "")
            clean = "".join(c for c in raw.lower() if c not in string.punctuation).strip()
            if clean in clean_profanity:
                # Bug fix 1a: handle missing word-level timestamps (fast/overlapping speech).
                # WhisperX sometimes omits start/end on individual words — fall back to
                # segment bounds rather than crashing or silently skipping the word.
                if "start" not in word or "end" not in word:
                    logger.warning(
                        f"Missing timestamps for word '{raw}' in segment "
                        f"{segment.get('start'):.2f}-{segment.get('end'):.2f} "
                        f"— using segment bounds as fallback."
                    )
                    # Don't add pad_after to segment["end"] — the segment end
                    # already includes natural duration and trailing silence.
                    # Extra padding here would cause the bleep to spill well
                    # beyond the actual word.
                    hits.append({
                        "start": max(0.0, float(segment["start"]) - pad_before),
                        "end":   float(segment["end"]),
                    })
                    continue

                raw_end = float(word["end"])

                # Bug fix 2: cap end time to prevent beep bleeding into post-word silence.
                # WhisperX often sets the last word's end = segment end (which includes
                # trailing silence). Cap to a generous max single-word duration (1.2s)
                # and to the start of the next word when available.
                max_word_duration = 1.2
                capped_end = min(raw_end, float(word["start"]) + max_word_duration)
                if i + 1 < len(words) and "start" in words[i + 1]:
                    capped_end = min(capped_end, float(words[i + 1]["start"]) - 0.05)

                hits.append({
                    "start": max(0.0, float(word["start"]) - pad_before),
                    "end":   capped_end + pad_after,
                })

    # Merge overlapping intervals — and also merge near-adjacent ones (< 0.15s gap).
    # Bug fix 1b: without this, two rapid consecutive swear words produce a tiny gap
    # in the mute window, letting a sliver of original audio bleed through.
    hits.sort(key=lambda x: x["start"])
    merged = []
    for h in hits:
        if merged and h["start"] - merged[-1]["end"] < 0.15:
            merged[-1]["end"] = max(merged[-1]["end"], h["end"])
        else:
            merged.append(dict(h))
    return merged


def get_non_profanity_intervals(profanity_ts: list[dict], duration: float) -> list[dict]:
    """Invert profanity timestamps → clean intervals."""
    if not profanity_ts:
        return [{"start": 0.0, "end": duration}]
    clean = []
    if profanity_ts[0]["start"] > 0:
        clean.append({"start": 0.0, "end": profanity_ts[0]["start"]})
    for i in range(len(profanity_ts) - 1):
        clean.append({"start": profanity_ts[i]["end"], "end": profanity_ts[i + 1]["start"]})
    if profanity_ts[-1]["end"] < duration:
        clean.append({"start": profanity_ts[-1]["end"], "end": duration})
    return clean


def _smooth_mute_envelope(profanity_ts: list[dict], fade: float) -> str:
    """
    Build a per-frame ffmpeg volume expression for the main audio (1 = full, 0 = muted).
    Smoothly ramps from 1 → 0 at each bleep start and 0 → 1 at each bleep end,
    giving a natural fade rather than a hard cut.
    Segments must be non-overlapping (call after merging).
    """
    if not profanity_ts:
        return "1"
    parts = []
    for b in profanity_ts:
        s, e = b["start"], b["end"]
        f  = min(fade, (e - s) / 4)   # cap at 25 % of segment so fade fits
        sf = s + f
        ef = max(sf, e - f)            # guard against tiny segments
        # Each segment contributes a dip envelope (0 outside, ramps 0→1 inside)
        parts.append(
            f"if(between(t,{s:.4f},{sf:.4f}),(t-{s:.4f})/{f:.4f},"
            f"if(between(t,{sf:.4f},{ef:.4f}),1,"
            f"if(between(t,{ef:.4f},{e:.4f}),({e:.4f}-t)/{f:.4f},0)))"
        )
    # mute factor = 1 − sum(dips); segments are non-overlapping so sum ≤ 1
    return f"1-({'+'.join(parts)})"


def _smooth_bleep_envelope(profanity_ts: list[dict], peak: float, fade: float) -> str:
    """
    Build a per-frame ffmpeg volume expression for the bleep tone.
    Ramps 0 → peak over `fade` s at start, holds, then ramps peak → 0 at end.
    """
    if not profanity_ts:
        return "0"
    parts = []
    for b in profanity_ts:
        s, e = b["start"], b["end"]
        f  = min(fade, (e - s) / 4)
        sf = s + f
        ef = max(sf, e - f)
        parts.append(
            f"if(between(t,{s:.4f},{sf:.4f}),(t-{s:.4f})/{f:.4f}*{peak},"
            f"if(between(t,{sf:.4f},{ef:.4f}),{peak},"
            f"if(between(t,{ef:.4f},{e:.4f}),({e:.4f}-t)/{f:.4f}*{peak},0)))"
        )
    return "+".join(parts)


def _build_ffmpeg_filter(profanity_ts: list[dict],
                          duration: float,
                          bleep_freq: int   = BLEEP_FREQUENCY,
                          bleep_vol: float  = BLEEP_VOLUME,
                          fade: float       = BLEEP_FADE_DURATION) -> str:
    """
    Build an ffmpeg filter_complex that:
      - Smoothly mutes the center channel around each profanity window
      - Mixes in a configurable sine tone that ramps in/out at each boundary
    The smooth envelopes (25 ms default) eliminate the abrupt hard-cut artefact.
    """
    mute_expr  = _smooth_mute_envelope(profanity_ts, fade)
    bleep_expr = _smooth_bleep_envelope(profanity_ts, bleep_vol, fade)

    mute_filter  = f"[0]volume='{mute_expr}':eval=frame[main]"
    bleep_filter = (
        f"sine=f={bleep_freq},"
        f"volume='{bleep_expr}':eval=frame,"
        f"aformat=channel_layouts=mono"
        f"[beep]"
    )
    amix = "[main][beep]amix=inputs=2:duration=first"
    return ";".join([mute_filter, bleep_filter, amix])


def replace_words_in_srt(lines: list[str], profanity: list[str]) -> list[str]:
    single_words = sorted(
        [w for w in profanity if " " not in w and '"' not in w], key=len, reverse=True
    )
    phrases = sorted(
        [p.strip('"') for p in profanity if " " in p or '"' in p], key=len, reverse=True
    )
    word_re   = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in single_words) + r")\b", re.IGNORECASE
    ) if single_words else None
    phrase_re = re.compile(
        "|".join(re.escape(p) for p in phrases), re.IGNORECASE
    ) if phrases else None

    result = []
    for line in lines:
        if not line.strip().isdigit() and "-->" not in line:
            if phrase_re:
                line = phrase_re.sub(lambda m: "*" * len(m.group()), line)
            if word_re:
                line = word_re.sub(lambda m: "*" * len(m.group()), line)
        result.append(line)
    return result


def apply_complex_filter_to_audio(center_path: str, filter_complex: str,
                                   bitrate=None, sample_rate=None,
                                   codec: str | None = None) -> str:
    base, ext  = os.path.splitext(os.path.basename(center_path))
    output_file = f"{base}_redacted{ext}"
    output_path = os.path.join(UPLOAD_FOLDER, output_file)

    cmd = (
        ["ffmpeg", "-y"]
        + _hwaccel_flags(video=False)
        + ["-i", center_path,
           "-filter_complex", filter_complex,
           "-bitexact", "-ac", "1",
           "-strict", "-2"]
    )
    if bitrate:   cmd += ["-b:a", str(bitrate)]
    if sample_rate: cmd += ["-ar", str(sample_rate)]
    if codec:     cmd += ["-c:a", codec]
    cmd += ["-max_interleave_delta", "0", output_path]

    _run(cmd, step="apply_bleep_filter")
    return output_file


def normalize_center_audio(modified_file: str, bitrate=None, sample_rate=None,
                            codec: str | None = None,
                            loudness_info: float | None = None) -> str:
    """
    Two-pass loudness normalization of the redacted center channel.

    Pass 1: measure actual loudness of the redacted file.
    Pass 2: apply loudnorm with measured values for accurate normalization.
    """
    modified_path = os.path.join(UPLOAD_FOLDER, modified_file)
    base, ext      = os.path.splitext(os.path.basename(modified_file))
    norm_out_file  = f"{base}_normalized{ext}"
    norm_out_path  = os.path.join(UPLOAD_FOLDER, norm_out_file)

    # Pass 1: measure the redacted file's actual loudness
    measure_cmd = [
        "ffmpeg", "-y",
        "-i", modified_path,
        "-filter:a", "loudnorm=print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(measure_cmd, capture_output=True, text=True)
    output = result.stderr or result.stdout

    # Use same robust JSON extraction as _measure_loudness
    measured = _extract_loudnorm_json(output)
    if measured:
        measured_i     = float(measured.get("input_i", -24.0))
        measured_lra   = float(measured.get("input_lra", 7.0))
        measured_tp    = float(measured.get("input_tp", -2.0))
        measured_thresh = float(measured.get("input_thresh", -34.0))
    else:
        logger.warning("Could not measure loudness of redacted audio – using defaults.")
        measured_i      = loudness_info if loudness_info is not None else -24.0
        measured_lra    = 7.0
        measured_tp     = -2.0
        measured_thresh = -34.0

    # Pass 2: apply loudnorm with measured values
    cmd = (
        ["ffmpeg", "-y"]
        + ["-i", modified_path,
           "-bitexact", "-ac", "1",
           "-strict", "-2",
           "-af", (
               f"loudnorm=I=-24:LRA=7:TP=-2:"
               f"measured_I={measured_i:.2f}:"
               f"measured_LRA={measured_lra:.2f}:"
               f"measured_TP={measured_tp:.2f}:"
               f"measured_thresh={measured_thresh:.2f}:"
               f"linear=true:print_format=summary"
           )]
    )
    if bitrate:     cmd += ["-b:a", str(bitrate)]
    if sample_rate: cmd += ["-ar", str(sample_rate)]
    if codec:       cmd += ["-c:a", codec]
    cmd += ["-max_interleave_delta", "0", norm_out_path]

    _run(cmd, step="normalize_center_audio")
    return norm_out_file


def redact_audio(job_id: str,
                  pad_before: float = 0.10,
                  pad_after:  float = 0.10) -> str:
    config = get_config(job_id)
    if not config:
        raise ValueError(f"Config not found for job_id: {job_id}")

    json_filename  = config.get("transcription_json")
    srt_filename   = config.get("transcription_srt")
    center_file    = config.get("center_channel_file")

    if not all([json_filename, srt_filename, center_file]):
        raise ValueError("Missing transcription_json, transcription_srt or center_channel_file")

    profanity   = _load_profanity_list()
    stream_info = (config.get("audio_stream_info", {}).get("streams") or [{}])[0]
    codec       = stream_info.get("codec_name") or config.get("source_codec", "ac3")
    bit_rate    = stream_info.get("bit_rate")   or config.get("source_bit_rate")
    sample_rate = stream_info.get("sample_rate") or config.get("source_sample_rate")
    channels    = int(stream_info.get("channels") or config.get("source_channels", 6))
    loudness    = config.get("loudness_info")

    json_path   = os.path.join(UPLOAD_FOLDER, json_filename)
    srt_path    = os.path.join(UPLOAD_FOLDER, srt_filename)
    center_path = os.path.join(UPLOAD_FOLDER, center_file)

    with open(json_path) as f:
        ts_data = json.load(f)

    profanity_ts   = identify_profanity_timestamps(ts_data, profanity,
                                                    pad_before=pad_before,
                                                    pad_after=pad_after)
    duration       = ts_data["segments"][-1]["end"] if ts_data.get("segments") else 0.0

    # Redact SRT (always — even if no profanity, produce the _redacted copy)
    with open(srt_path, encoding="utf-8") as srt_fh:
        srt_lines = srt_fh.readlines()
    modified_lines = replace_words_in_srt(srt_lines, profanity)
    srt_base       = os.path.splitext(srt_filename)[0]
    out_srt_file   = f"{srt_base}_redacted_subtitle.srt"
    out_srt_path   = os.path.join(UPLOAD_FOLDER, out_srt_file)
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.writelines(modified_lines)
    update_config(job_id, {"redacted_srt": out_srt_file})

    if not profanity_ts:
        # No profanity in FC — use center channel as-is
        logger.info("No profanity detected in FC – skipping FC bleep filter.")
        if channels <= 2:
            update_config(job_id, {"redacted_audio_stream_final": center_file})
            logger.info("Redaction complete (no profanity found).")
            return "Redaction completed successfully (no profanity)"
        else:
            update_config(job_id, {"redacted_channel_FC": center_file})
            _redact_flr_pass(job_id, config, profanity, duration, bit_rate, sample_rate, codec, pad_before, pad_after)
            logger.info("Redaction complete.")
            return "Redaction completed successfully"

    clean_intervals = get_non_profanity_intervals(profanity_ts, duration)  # noqa: F841
    filter_complex  = _build_ffmpeg_filter(profanity_ts, duration)

    # Apply bleep filter to center channel
    modified_center = apply_complex_filter_to_audio(
        center_path, filter_complex, bit_rate, sample_rate, codec
    )

    if channels <= 2:
        # Mono/stereo: redacted center IS the final audio
        update_config(job_id, {"redacted_audio_stream_final": modified_center})
    else:
        # Multichannel: normalize then recombine
        normalized_center = normalize_center_audio(
            modified_center, bit_rate, sample_rate, codec, loudness
        )
        update_config(job_id, {"redacted_channel_FC": normalized_center})
        _redact_flr_pass(job_id, config, profanity, duration, bit_rate, sample_rate, codec, pad_before, pad_after)

    logger.info("Redaction complete.")
    return "Redaction completed successfully"


def _redact_flr_pass(job_id: str, config: dict, profanity: list[str],
                     fc_duration: float, bit_rate, sample_rate, codec: str,
                     pad_before: float, pad_after: float) -> None:
    """Apply bleep filter to FL+FR stereo channel if FL+FR transcription exists."""
    flr_json_filename = config.get("flr_transcription_json")
    flr_file          = config.get("flr_channel_file")
    if not (flr_json_filename and flr_file):
        return

    flr_json_path = os.path.join(UPLOAD_FOLDER, flr_json_filename)
    flr_path      = os.path.join(UPLOAD_FOLDER, flr_file)
    if not (os.path.exists(flr_json_path) and os.path.exists(flr_path)):
        logger.warning("FL+FR transcription or audio file missing — skipping FL+FR redaction.")
        return

    with open(flr_json_path) as f:
        flr_ts_data = json.load(f)

    flr_profanity_ts = identify_profanity_timestamps(
        flr_ts_data, profanity, pad_before=pad_before, pad_after=pad_after
    )
    if not flr_profanity_ts:
        logger.info("FL+FR: no profanity detected — original FL+FR preserved.")
        return

    logger.info(f"FL+FR: {len(flr_profanity_ts)} profanity hit(s) — bleeping.")
    flr_duration = flr_ts_data["segments"][-1]["end"] if flr_ts_data.get("segments") else fc_duration
    flr_filter   = _build_ffmpeg_filter(flr_profanity_ts, flr_duration)
    modified_flr = apply_complex_filter_to_audio(flr_path, flr_filter, bit_rate, sample_rate, codec)
    update_config(job_id, {"redacted_channel_FLR": modified_flr})
    logger.info(f"FL+FR redacted → {modified_flr}")


# ---------------------------------------------------------------------------
# Section 9 – Stage 6: Combine media file
# Dynamically handles any channel layout – no more hardcoded 5.1(side).
# ---------------------------------------------------------------------------

def _build_pan_filter(layout: str, fc_input_index: int, original_input_index: int,
                      flr_input_index: int | None = None) -> str:
    """
    Build the ffmpeg filter_complex to replace FC (and optionally FL+FR) in the final mix.

    With flr_input_index=None  (FC only):
        amerge=inputs=2  → original(n ch) + fc_mono(1 ch)

    With flr_input_index set  (FC + FL+FR):
        amerge=inputs=3  → original(n ch) + fc_mono(1 ch) + flr_stereo(2 ch)
        pan replaces FC with c(n), FL with c(n+1), FR with c(n+2)
    """
    ch_list = CHANNEL_LAYOUTS.get(layout)
    if not ch_list:
        logger.warning(f"Unknown layout {layout!r} – using passthrough.")
        return f"[{original_input_index}:a]acopy[final]"

    n = len(ch_list)

    if "FC" not in ch_list:
        return f"[{original_input_index}:a]acopy[final]"

    if flr_input_index is not None and "FL" in ch_list and "FR" in ch_list:
        # 3-input merge: original + fc_mono + flr_stereo
        channel_map = "|".join(
            f"c{i}=c{n}"     if ch == "FC" else
            f"c{i}=c{n+1}"   if ch == "FL" else
            f"c{i}=c{n+2}"   if ch == "FR" else
            f"c{i}=c{i}"
            for i, ch in enumerate(ch_list)
        )
        return (
            f"[{fc_input_index}:a]aformat=channel_layouts=mono[fc_mono];"
            f"[{flr_input_index}:a]aformat=channel_layouts=stereo[flr_stereo];"
            f"[{original_input_index}:a][fc_mono][flr_stereo]amerge=inputs=3[merged];"
            f"[merged]pan={layout}|{channel_map}[final]"
        )
    else:
        # 2-input merge: original + fc_mono only
        channel_map = "|".join(
            f"c{i}=c{n}" if ch == "FC" else f"c{i}=c{i}"
            for i, ch in enumerate(ch_list)
        )
        return (
            f"[{fc_input_index}:a]aformat=channel_layouts=mono[fc_mono];"
            f"[{original_input_index}:a][fc_mono]amerge=inputs=2[merged];"
            f"[merged]pan={layout}|{channel_map}[final]"
        )


def combine_media_file(job_id: str) -> str:
    config = get_config(job_id)
    if not config:
        raise ValueError(f"Config not found for job_id: {job_id}")

    input_media_file = config.get("input_filename")
    redacted_srt     = config.get("redacted_srt")
    channels         = int(config.get("source_channels", 6))
    layout           = config.get("source_layout", "5.1")
    codec            = config.get("source_codec", "ac3")
    bit_rate         = config.get("source_bit_rate")
    sample_rate      = config.get("source_sample_rate")
    orig_stream_idx  = config.get("audio_stream_index_nr", 0)

    if channels <= 2:
        redacted_audio = config.get("redacted_audio_stream_final")
    else:
        redacted_audio = config.get("redacted_channel_FC")

    redacted_flr       = config.get("redacted_channel_FLR")
    redacted_flr_path  = os.path.join(UPLOAD_FOLDER, redacted_flr) if redacted_flr else None
    use_flr            = bool(redacted_flr_path and os.path.exists(redacted_flr_path))

    if not all([input_media_file, redacted_audio, redacted_srt]):
        raise ValueError("Missing input_filename, redacted_audio, or redacted_srt in config")

    base, ext = os.path.splitext(os.path.basename(input_media_file))
    output_file = f"{base}_final{ext}"

    input_media_path    = config.get("input_filepath") or os.path.join(UPLOAD_FOLDER, input_media_file)
    redacted_audio_path = os.path.join(UPLOAD_FOLDER, redacted_audio)
    redacted_srt_path   = os.path.join(UPLOAD_FOLDER, redacted_srt)
    output_path         = os.path.join(UPLOAD_FOLDER, output_file)

    # Inspect existing subtitle streams — strip MOV/tx3g, keep SRT/ASS/PGS
    MOV_SUB_CODECS = {"mov_text", "tx3g", "mp4s"}
    try:
        probe = _probe(input_media_path)
        all_streams   = probe.get("streams", [])
        sub_streams   = [s for s in all_streams if s.get("codec_type") == "subtitle"]
        keep_sub_idxs = [i for i, s in enumerate(sub_streams)
                         if s.get("codec_name", "").lower() not in MOV_SUB_CODECS]
        redacted_sub_idx = len(keep_sub_idxs)
    except Exception:
        sub_streams      = []
        keep_sub_idxs    = []
        redacted_sub_idx = 0

    if channels <= 2:
        cmd = (
            ["ffmpeg", "-y"]
            + _hwaccel_flags()
            + [
                "-i", input_media_path,
                "-i", redacted_audio_path,
                "-i", redacted_srt_path,
                "-map", "0:v",
                "-map", "1:a",
                "-map", f"0:a:{orig_stream_idx}",
            ]
            + [arg for idx in keep_sub_idxs for arg in ["-map", f"0:s:{idx}"]]
            + ["-map", "2:s",
               "-c:v", "copy",
               "-c:a:0", codec,
               "-c:a:1", "copy",
            ]
        )
    else:
        audio_file  = config.get("audio_filename")
        audio_path  = os.path.join(UPLOAD_FOLDER, audio_file) if audio_file else None

        if audio_path and os.path.exists(audio_path):
            # Inputs: 0=MKV, 1=extracted audio, 2=redacted FC, [3=redacted FLR], N=SRT
            extra_inputs  = ["-i", redacted_flr_path] if use_flr else []
            srt_input_idx = 3 + int(use_flr)
            flr_idx       = 3 if use_flr else None
            cmd = (
                ["ffmpeg", "-y"]
                + _hwaccel_flags()
                + [
                    "-i", input_media_path,
                    "-i", audio_path,
                    "-i", redacted_audio_path,
                ]
                + extra_inputs
                + ["-i", redacted_srt_path,
                   "-filter_complex",
                   _build_pan_filter(layout, fc_input_index=2, original_input_index=1,
                                     flr_input_index=flr_idx),
                   "-map", "0:v",
                   "-map", "[final]",
                   "-map", f"0:a:{orig_stream_idx}",
                ]
                + [arg for idx in keep_sub_idxs for arg in ["-map", f"0:s:{idx}"]]
                + [f"-map", f"{srt_input_idx}:s",
                   "-c:v", "copy",
                   "-c:a:0", codec,
                ]
            )
        else:
            cmd = (
                ["ffmpeg", "-y"]
                + _hwaccel_flags()
                + [
                    "-i", input_media_path,
                    "-i", redacted_audio_path,
                ]
                + (["-i", redacted_flr_path] if use_flr else [])
                + ["-i", redacted_srt_path,
                   "-filter_complex",
                   _build_pan_filter(layout, fc_input_index=1, original_input_index=0,
                                     flr_input_index=2 if use_flr else None),
                   "-map", "0:v",
                   "-map", "[final]",
                   "-map", f"0:a:{orig_stream_idx}",
                ]
                + [arg for idx in keep_sub_idxs for arg in ["-map", f"0:s:{idx}"]]
                + ["-map", f"{2 + int(use_flr)}:s",
                   "-c:v", "copy",
                   "-c:a:0", codec,
                ]
            )

    if bit_rate:    cmd += ["-b:a:0", str(bit_rate)]
    if sample_rate: cmd += ["-ar", str(sample_rate)]

    cmd += ["-c:a:1", "copy"]

    # MP4/MOV containers only support mov_text subtitles — subrip/srt will error.
    # MKV and others accept srt/copy directly.
    mp4_container = os.path.splitext(output_path)[1].lower() in (".mp4", ".m4v", ".mov")
    sub_codec_copy     = "mov_text" if mp4_container else "copy"
    sub_codec_redacted = "mov_text" if mp4_container else "srt"

    cmd += ["-c:s", sub_codec_copy]
    cmd += [f"-c:s:{redacted_sub_idx}", sub_codec_redacted]

    cmd += [
        "-metadata:s:a:0", "title=Family audio",
        "-metadata:s:a:0", "language=eng",
        "-disposition:a:0", "default",               # family audio = default
        "-metadata:s:a:1", "title=Original audio",
        "-metadata:s:a:1", "language=eng",
        "-disposition:a:1", "0",
        f"-metadata:s:s:{redacted_sub_idx}", "title=Redacted subtitles",
        f"-metadata:s:s:{redacted_sub_idx}", "language=eng",
        f"-disposition:s:{redacted_sub_idx}", "default",   # redacted subs = default
        "-strict", "-2",
        output_path,
    ]

    _run(cmd, step="combine_media")

    update_config(job_id, {"final_output": output_file})
    logger.info(f"Combined media → {output_file}")
    return "Combine media completed successfully"


# ---------------------------------------------------------------------------
# Section 10 – Stage 7: Cleanup
# ---------------------------------------------------------------------------

def cleanup_job_files(job_id: str) -> str:
    config = get_config(job_id)
    if not config:
        raise ValueError(f"Config not found for job_id: {job_id}")

    original_filename = config.get("original_filename")
    input_filename    = config.get("input_filename")
    final_output      = config.get("final_output")

    if not final_output:
        raise ValueError("final_output not set in config – nothing to clean up.")

    # Use original filename as-is — secure_filename strips { } [ ] which
    # are valid in arr-generated filenames. Path traversal is prevented
    # by the realpath check below.
    safe_original = os.path.basename(original_filename) if original_filename else None
    if not safe_original:
        raise ValueError("original_filename is missing or invalid in config.")

    final_path = os.path.join(UPLOAD_FOLDER, final_output)
    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Final output not found: {final_path}")

    # If the file came from an absolute path on the media volume,
    # move the output back there (overwrite original with bleeped version).
    # Otherwise fall back to placing it in the uploads folder.
    input_filepath = config.get("input_filepath")
    if input_filepath:
        dest_path = input_filepath
    else:
        dest_path = os.path.join(UPLOAD_FOLDER, safe_original)
        if not os.path.realpath(dest_path).startswith(os.path.realpath(UPLOAD_FOLDER)):
            raise ValueError("original_filename path escapes upload folder.")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        os.replace(final_path, dest_path)
        logger.info(f"Moved (same device) {final_output} → {dest_path}")
    except OSError as e:
        if e.errno == 18:  # cross-device link — fall back to copy + delete
            logger.info(f"Cross-device move detected, copying {final_output} → {dest_path}")
            shutil.copy2(final_path, dest_path)
            os.remove(final_path)
            logger.info(f"Copy complete, removed source {final_path}")
        else:
            raise

    base = os.path.splitext(input_filename)[0]
    patterns = [
        f"{job_id}_*",            # config + any job-tagged files
        f"{base}*center*.*",      # center channel extractions (e.g. _norm_center.ac3)
        f"{base}*flr*.*",         # FL+FR stereo extractions
        f"{base}_audio*.*",
        f"{base}_norm*.*",
        f"{base}_*redacted*.*",
        f"{base}_*normalized*.*",
        f"{base}_*final*.*",
        f"{base}*.json",          # transcription JSON
        f"{base}*.srt",           # raw + redacted SRT
        f"{base}*.vtt",           # WebVTT subtitles
        f"{base}*.txt",           # plain-text transcript
        f"{base}*.ac3",           # normalized + extracted audio
        f"{base}*.dts",
        f"{base}*.aac",
        f"{base}*.flac",
        f"{base}*.wav",
    ]
    removed = []
    for pattern in patterns:
        for f in glob.glob(os.path.join(UPLOAD_FOLDER, pattern)):
            if os.path.isfile(f) and os.path.abspath(f) != os.path.abspath(dest_path):
                os.remove(f)
                removed.append(os.path.basename(f))

    if removed:
        logger.info(f"Cleanup removed {len(removed)} temp files: {', '.join(removed)}")
    logger.info(f"Cleanup done for job {job_id}.")
    return dest_path


# ---------------------------------------------------------------------------
# Section 11 – Full pipeline (fire-and-forget)
# ---------------------------------------------------------------------------

def run_full_pipeline(job_id: str, whisperx_settings: dict | None = None,
                       pad_before: float = 0.10,
                       pad_after:  float = 0.10,
                       plex_url: str | None = None,
                       plex_token: str | None = None,
                       plex_section_id: str | None = None) -> None:
    """
    Run all pipeline stages sequentially in a background thread.
    Updates job_status at each stage.  On completion, optionally triggers
    a Plex library refresh.
    """
    import requests  # local import – only needed here

    def _update(status: str, stage: str = "") -> None:
        _set_job_status(job_id, status)
        update_config(job_id, {"pipeline_status": status, "pipeline_stage": stage})
        logger.info(f"[pipeline:{job_id}] {stage} → {status}")

    def _already_done(stage: str) -> bool:
        """Return True if this stage already completed in a prior run."""
        config = get_config(job_id) or {}
        completed = config.get("completed_stages", [])
        if stage in completed:
            return True
        # Fallback: check if the expected output file already exists
        file_checks = {
            "analyze_audio":  lambda c: c.get("audio_stream_index_nr") is not None,
            "normalize_audio": lambda c: c.get("normalization_skipped") or (
                c.get("normalized_audio_file") and
                os.path.exists(os.path.join(UPLOAD_FOLDER, c.get("normalized_audio_file", "")))),
            "extract_audio":  lambda c: c.get("center_channel_file") and
                os.path.exists(os.path.join(UPLOAD_FOLDER, c.get("center_channel_file", ""))),
            "transcribe":     lambda c: c.get("transcription_srt") and
                os.path.exists(os.path.join(UPLOAD_FOLDER, c.get("transcription_srt", ""))),
            "redact":         lambda c: c.get("redacted_srt") and
                os.path.exists(os.path.join(UPLOAD_FOLDER, c.get("redacted_srt", ""))),
            "combine":        lambda c: c.get("final_output") and
                os.path.exists(os.path.join(UPLOAD_FOLDER, c.get("final_output", ""))),
        }
        check = file_checks.get(stage)
        if check and check(config):
            logger.info(f"[pipeline:{job_id}] {stage} → output exists, marking done")
            _mark_done(stage)
            return True
        return False

    def _mark_done(stage: str) -> None:
        config = get_config(job_id) or {}
        completed = config.get("completed_stages", [])
        if stage not in completed:
            completed.append(stage)
        update_config(job_id, {"completed_stages": completed})

    try:
        if not _already_done("analyze_audio"):
            _update("running", "analyze_audio")
            analyze_and_select_audio_stream(job_id)
            _mark_done("analyze_audio")
        else:
            logger.info(f"[pipeline:{job_id}] analyze_audio → skipped (already done)")

        if not _already_done("normalize_audio"):
            _update("running", "normalize_audio")
            normalize_audio_stream(job_id)
            _mark_done("normalize_audio")
        else:
            logger.info(f"[pipeline:{job_id}] normalize_audio → skipped (already done)")

        if not _already_done("extract_audio"):
            _update("running", "extract_audio")
            extract_audio_stream(job_id)
            _mark_done("extract_audio")
        else:
            logger.info(f"[pipeline:{job_id}] extract_audio → skipped (already done)")

        if not _already_done("transcribe"):
            _update("running", "transcribe")
            transcribe_audio(job_id, whisperx_settings)
            _mark_done("transcribe")
        else:
            logger.info(f"[pipeline:{job_id}] transcribe → skipped (already done)")

        if not _already_done("redact"):
            _update("running", "redact")
            redact_audio(job_id, pad_before=pad_before, pad_after=pad_after)
            _mark_done("redact")
        else:
            logger.info(f"[pipeline:{job_id}] redact → skipped (already done)")

        if not _already_done("combine"):
            _update("running", "combine")
            combine_media_file(job_id)
            _mark_done("combine")
        else:
            logger.info(f"[pipeline:{job_id}] combine → skipped (already done)")

        _update("running", "cleanup")
        cleanup_job_files(job_id)

        _update("completed", "done")
        logger.info(f"[pipeline:{job_id}] ✅ Pipeline completed successfully.")

        # Optional Plex refresh
        # plex_section_id can be a single ID ("1"), comma-separated ("1,2"),
        # or empty/None to refresh all sections.
        if plex_url and plex_token:
            try:
                base = plex_url.rstrip('/')
                token_param = f"?X-Plex-Token={plex_token}"
                if plex_section_id:
                    section_ids = [s.strip() for s in str(plex_section_id).split(",") if s.strip()]
                else:
                    section_ids = []

                if section_ids:
                    for sid in section_ids:
                        resp = requests.get(f"{base}/library/sections/{sid}/refresh{token_param}", timeout=10)
                        logger.info(f"Plex refresh section {sid} → HTTP {resp.status_code}")
                else:
                    # Refresh all sections
                    resp = requests.get(f"{base}/library/sections/all/refresh{token_param}", timeout=10)
                    logger.info(f"Plex refresh all sections → HTTP {resp.status_code}")
            except Exception as plex_err:
                logger.warning(f"Plex refresh failed (non-fatal): {plex_err}")

    except Exception as e:
        _update("failed", "error")
        update_config(job_id, {"pipeline_error": str(e), "pipeline_traceback": traceback.format_exc()})
        logger.error(f"[pipeline:{job_id}] ❌ Failed: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Section 12 – Flask routes
# ---------------------------------------------------------------------------

def _job_id_from_request() -> str:
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id") or request.args.get("job_id", "")
    if not job_id:
        raise ValueError("job_id is required")
    return job_id


def _require_config(job_id: str) -> dict:
    config = get_config(job_id)
    if not config:
        raise ValueError(f"No config found for job_id: {job_id}")
    return config


def _error(msg: str, code: int = 500, job_id: str = "", details: str = "") -> tuple:
    body = {"status": "error", "message": msg}
    if job_id:   body["job_id"] = job_id
    if details:  body["details"] = details
    return jsonify(body), code


# --- Utility ---

@app.route("/api/list_files", methods=["GET"])
def list_files():
    files = [
        f for f in os.listdir(UPLOAD_FOLDER)
        if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)) and allowed_file(f)
    ]
    return jsonify({"files": files})


@app.route("/api/job_status/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    config = get_config(job_id)
    if not config:
        return _error("Job not found", 404, job_id)
    with _pipeline_queue_lock:
        queue_position = (_pipeline_queue_order.index(job_id) + 1
                          if job_id in _pipeline_queue_order else None)
    return jsonify({
        "job_id":         job_id,
        "status":         _get_job_status(job_id) if _get_job_status(job_id) != "unknown" else config.get("pipeline_status", "unknown"),
        "stage":          config.get("pipeline_stage", ""),
        "error":          config.get("pipeline_error", ""),
        "queue_position": queue_position,   # None = running or done, N = waiting
        "queue_depth":    _pipeline_queue.qsize(),
    })


# --- Job lifecycle ---

@app.route("/api/initialize_job", methods=["POST"])
def initialize_job():
    job_id = str(uuid.uuid4())
    update_config(job_id, {"job_id": job_id})
    _set_job_status(job_id, "initialized")
    logger.info(f"Initialized job {job_id}")
    return jsonify({"job_id": job_id})


@app.route("/api/select_remote_file", methods=["POST"])
def select_remote_file():
    try:
        job_id   = _job_id_from_request()
        data     = request.get_json()
        filename = data.get("filename", "")
        if not filename:
            return _error("filename is required", 400)

        safe_name = secure_filename(filename)
        if not safe_name:
            return _error("Invalid filename", 400)
        file_path = os.path.join(UPLOAD_FOLDER, safe_name)

        # Validate resolved path stays within UPLOAD_FOLDER
        if not os.path.realpath(file_path).startswith(os.path.realpath(UPLOAD_FOLDER)):
            return _error("Invalid file path", 400)

        if not os.path.isfile(file_path):
            return _error(f"File not found: {safe_name}", 404)

        try:
            mime      = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            if not (file_type.startswith("video/") or file_type.startswith("audio/")):
                return _error(f"File is not a video/audio: {file_type}", 400)
        except Exception as e:
            logger.warning(f"MIME check failed: {e} – proceeding anyway.")

        if not allowed_file(safe_name):
            return _error(f"Extension not allowed: {safe_name}", 400)

        update_config(job_id, {"original_filename": safe_name, "input_filename": safe_name})
        return jsonify({"remote_file": filename, "job_id": job_id})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return _error("No file in request", 400)

        file   = request.files["file"]
        job_id = request.form.get("job_id", "")
        if not job_id:
            return _error("job_id required", 400)
        if not file.filename:
            return _error("No file selected", 400)
        if not allowed_file(file.filename):
            return _error(f"File type not allowed: {file.filename}", 400)

        filename    = secure_filename(f"{job_id}_input_{file.filename}")
        upload_path = os.path.join(UPLOAD_FOLDER, filename)

        update_config(job_id, {"original_filename": file.filename})

        with open(upload_path, "wb") as f:
            while True:
                chunk = file.read(app.config["CHUNK_SIZE"])
                if not chunk:
                    break
                f.write(chunk)

        update_config(job_id, {"input_filename": filename})
        logger.info(f"Uploaded {filename} for job {job_id}")
        return jsonify({"job_id": job_id, "filename": filename}), 200
    except Exception as e:
        return _error(str(e), 500)


# --- Pipeline stages ---

@app.route("/api/analyze_and_select_audio", methods=["POST"])
def api_analyze_and_select_audio():
    try:
        job_id = _job_id_from_request()
        _require_config(job_id)
        selected = analyze_and_select_audio_stream(job_id)
        return jsonify({"status": "success", "selected_stream": selected})
    except Exception as e:
        return _error(str(e), 500, details=traceback.format_exc())


@app.route("/api/normalize_audio", methods=["POST"])
def api_normalize_audio():
    try:
        job_id = _job_id_from_request()
        data   = request.get_json(silent=True) or {}
        force  = data.get("force", False)
        _require_config(job_id)
        result = normalize_audio_stream(job_id, force=force)
        return jsonify({
            "status":  "success",
            "normalized_file": result,
            "skipped": result is None,
        })
    except Exception as e:
        return _error(str(e), 500, details=traceback.format_exc())


@app.route("/api/extract_audio", methods=["POST"])
def api_extract_audio():
    try:
        job_id = _job_id_from_request()
        _require_config(job_id)
        result = extract_audio_stream(job_id)
        return jsonify({"status": "success", **result})
    except Exception as e:
        return _error(str(e), 500, details=traceback.format_exc())


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    try:
        job_id   = _job_id_from_request()
        _require_config(job_id)
        settings = (request.get_json(silent=True) or {}).get("whisperx_settings")
        transcribe_audio(job_id, settings)
        return jsonify({"status": "success"})
    except Exception as e:
        return _error(str(e), 500, details=traceback.format_exc())


@app.route("/api/redact", methods=["POST"])
def api_redact():
    try:
        job_id = _job_id_from_request()
        data   = request.get_json(silent=True) or {}
        pad_before = float(data.get("pad_before", 0.10))
        pad_after  = float(data.get("pad_after",  0.10))
        _require_config(job_id)
        redact_audio(job_id, pad_before=pad_before, pad_after=pad_after)
        return jsonify({"status": "success"})
    except Exception as e:
        return _error(str(e), 500, details=traceback.format_exc())


@app.route("/api/combine_media", methods=["POST"])
def api_combine_media():
    try:
        job_id = _job_id_from_request()
        _require_config(job_id)
        combine_media_file(job_id)
        return jsonify({"status": "success"})
    except Exception as e:
        return _error(str(e), 500, details=traceback.format_exc())


@app.route("/api/cleanup", methods=["POST"])
def api_cleanup():
    try:
        job_id = _job_id_from_request()
        _require_config(job_id)
        final = cleanup_job_files(job_id)
        return jsonify({"status": "success", "final_filename": final})
    except Exception as e:
        return _error(str(e), 500, details=traceback.format_exc())


# --- Full pipeline (fire-and-forget) ---

@app.route("/api/process_full", methods=["POST"])
def api_process_full():
    """
    Fire-and-forget endpoint.  Kicks off the full pipeline in a background
    thread and returns immediately with the job_id.

    POST body (all optional except filename/job_id):
    {
        "job_id":           "<existing job_id>",   // if omitted, a new job is created
        "filename":         "movie.mkv",           // required if no job_id pre-loaded
        "whisperx_settings": {              // transcription settings (Parakeet TDT)
            // Model variants:
            "model": "nvidia/parakeet-tdt-0.6b-v3",  // default — faster, lower VRAM
            // "model": "nvidia/parakeet-tdt-1.1b",  // larger — slightly higher accuracy
        },
        "plex_url":         "http://plex:32400",
        "plex_token":       "abc123",
        "plex_section_id":  "1"
    }
    """
    try:
        data     = request.get_json(silent=True) or {}
        job_id   = data.get("job_id")
        filename = data.get("filename")

        # Resume existing incomplete job for this filename if one exists
        if filename and not job_id:
            existing = find_job_by_filename(filename)
            if existing:
                job_id = existing["job_id"]
                last_stage = existing.get("pipeline_stage", "")
                logger.info(f"Resuming existing job {job_id} for {os.path.basename(filename)} (last stage: {last_stage})")
            else:
                job_id = str(uuid.uuid4())
                update_config(job_id, {"job_id": job_id})
        elif not job_id:
            job_id = str(uuid.uuid4())
            update_config(job_id, {"job_id": job_id})

        _set_job_status(job_id, "queued")

        # Associate file if provided
        if filename:
            # If an absolute path is provided and it exists, use it directly
            # (file lives on the shared media volume, no copy needed)
            if os.path.isabs(filename) and os.path.exists(filename):
                safe = os.path.basename(filename)
                update_config(job_id, {
                    "original_filename": safe,
                    "input_filename":    safe,
                    "input_filepath":    filename,   # full absolute path
                })
            else:
                # Filename only — look in uploads folder (legacy / upload flow)
                # Use the original filename as-is (don't sanitize — arr filenames
                # contain brackets/spaces that secure_filename would strip)
                safe = os.path.basename(filename)
                file_path = os.path.join(UPLOAD_FOLDER, safe)
                # Prevent path traversal
                if not os.path.realpath(file_path).startswith(os.path.realpath(UPLOAD_FOLDER)):
                    return _error("Invalid file path", 400, job_id)
                if not os.path.exists(file_path):
                    return _error(f"File not found in uploads: {safe}", 404, job_id)
                update_config(job_id, {
                    "original_filename": safe,
                    "input_filename":    safe,
                })
        else:
            config = get_config(job_id)
            if not config or not config.get("input_filename"):
                return _error(
                    "Provide 'filename' or a pre-loaded job_id with input_filename set.",
                    400, job_id
                )

        whisperx_settings = data.get("whisperx_settings")
        pad_before        = float(data.get("pad_before", 0.10))
        pad_after         = float(data.get("pad_after",  0.10))
        plex_url          = data.get("plex_url")
        plex_token        = data.get("plex_token")
        plex_section_id   = data.get("plex_section_id")

        position = _queue_pipeline(
            job_id,
            run_full_pipeline,
            (job_id, whisperx_settings, pad_before, pad_after, plex_url, plex_token, plex_section_id),
        )

        msg = "Pipeline started." if position == 1 else f"Queued at position {position} — will start after current job finishes."
        return jsonify({
            "status":   "queued",
            "job_id":   job_id,
            "position": position,
            "message":  f"{msg} Poll /api/job_status/<job_id> for progress.",
        }), 202

    except Exception as e:
        return _error(str(e), 500, details=traceback.format_exc())
