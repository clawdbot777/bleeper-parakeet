#!/usr/bin/env python3
"""
Parakeet TDT transcription — outputs WhisperX-compatible JSON + SRT.

Usage:
    python parakeet_transcribe.py <audio_file> \
        --output_dir <dir> \
        [--model nvidia/parakeet-tdt-0.6b-v3]

Output format matches WhisperX exactly so the rest of the bleeper pipeline
doesn't need to know which backend was used:
  {
    "segments": [
      {
        "start": 1.04,
        "end":   3.21,
        "text":  "Hello world",
        "words": [
          {"word": "Hello", "start": 1.04, "end": 1.44},
          {"word": "world", "start": 1.60, "end": 2.10}
        ]
      },
      ...
    ]
  }

Parakeet requires 16 kHz mono WAV/FLAC.  Any other format (AC3, DTS, AAC…)
is automatically converted via ffmpeg before transcription and cleaned up
afterwards.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_srt_time(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = int(s % 60)
    ms  = int(round((s % 1) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _build_srt(segments: list[dict]) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        start = _to_srt_time(seg["start"])
        end   = _to_srt_time(seg["end"])
        text  = seg["text"].strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def _convert_to_wav(src: str) -> tuple[str, bool]:
    """
    Convert *src* to a 16 kHz mono WAV file.
    Returns (wav_path, needs_cleanup).
    If *src* is already a suitable WAV we return it as-is (no cleanup needed).
    """
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,sample_rate,channels",
             "-of", "json", src],
            capture_output=True, text=True, check=True,
        )
        info  = json.loads(probe.stdout)
        ainfo = (info.get("streams") or [{}])[0]
        if (src.lower().endswith(".wav")
                and ainfo.get("sample_rate") == "16000"
                and ainfo.get("channels") == 1):
            return src, False
    except Exception:
        pass  # fall through to conversion

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", src,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
         tmp.name],
        check=True, capture_output=True,
    )
    return tmp.name, True


def _words_in_segment(word_stamps: list[dict],
                       seg_start: float,
                       seg_end: float) -> list[dict]:
    """Return all word stamps whose midpoint falls within [seg_start, seg_end]."""
    result = []
    for w in word_stamps:
        mid = (w["start"] + w["end"]) / 2
        if seg_start - 0.02 <= mid <= seg_end + 0.02:
            result.append({"word": w["word"],
                           "start": w["start"],
                           "end":   w["end"]})
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with NVIDIA Parakeet TDT and write "
                    "WhisperX-compatible JSON + SRT."
    )
    parser.add_argument("audio",        help="Path to the audio file to transcribe.")
    parser.add_argument("--output_dir", default=".", help="Directory for output files.")
    parser.add_argument("--model",      default="nvidia/parakeet-tdt-0.6b-v3",
                        help="HuggingFace model name (default: parakeet-tdt-0.6b-v3).")
    parser.add_argument("--vad_filter", action="store_true",
                        help="Request VAD pre-filtering (not yet supported — logged as warning).")
    args = parser.parse_args()

    if args.vad_filter:
        print("[parakeet] WARNING: --vad_filter requested but not yet supported by Parakeet backend. "
              "Transcribing full stream.", file=sys.stderr, flush=True)

    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Convert audio to 16 kHz mono WAV if needed ---
    wav_path, cleanup_wav = _convert_to_wav(args.audio)
    try:
        # --- 2. Load model ---
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
        except ImportError:
            print(
                "ERROR: nemo_toolkit not installed.\n"
                "       Run: pip install nemo_toolkit[asr]",
                file=sys.stderr,
            )
            sys.exit(2)

        print(f"[parakeet] Loading model: {args.model}", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)

        # --- 3. Transcribe with timestamps ---
        print(f"[parakeet] Transcribing: {args.audio}", flush=True)

        word_stamps    = []
        segment_stamps = []

        try:
            # NeMo 2.1+ API — timestamps=True keyword supported directly
            outputs = model.transcribe([wav_path], timestamps=True)
            result  = outputs[0]
            word_stamps    = result.timestamp.get("word",    [])
            segment_stamps = result.timestamp.get("segment", [])
            print("[parakeet] Transcription complete (timestamps=True API).", flush=True)

        except TypeError:
            # NeMo 2.0.x (24.09 containers) — enable timestamps via decoding config
            print("[parakeet] timestamps=True not supported — configuring decoding for timestamps.",
                  file=sys.stderr, flush=True)
            try:
                from omegaconf import OmegaConf
                decoding_cfg = model.cfg.decoding
                with OmegaConf.open_dict(decoding_cfg):
                    decoding_cfg.preserve_alignments = True
                    decoding_cfg.compute_timestamps  = True
                model.change_decoding_strategy(decoding_cfg)
            except Exception as cfg_err:
                print(f"[parakeet] WARNING: could not configure timestamps via decoding_cfg: {cfg_err}",
                      file=sys.stderr, flush=True)

            outputs = model.transcribe([wav_path], return_hypotheses=True)
            result  = outputs[0]
            ts = getattr(result, 'timestamp', None) or {}
            word_stamps    = ts.get("word",    [])
            segment_stamps = ts.get("segment", [])
            print(f"[parakeet] Transcription complete (return_hypotheses API). "
                  f"words={len(word_stamps)}, segments={len(segment_stamps)}", flush=True)
        # --- 4. Normalise to WhisperX JSON schema ---
        segments = []
        for seg in segment_stamps:
            seg_start = float(seg["start"])
            seg_end   = float(seg["end"])
            seg_text  = seg.get("segment") or seg.get("text") or ""
            segments.append({
                "start": seg_start,
                "end":   seg_end,
                "text":  seg_text,
                "words": _words_in_segment(word_stamps, seg_start, seg_end),
            })

        # --- Fallback: no segment timestamps but have word timestamps ---
        if not segments and word_stamps:
            print("[parakeet] WARNING: no segment timestamps — building from word timestamps.",
                  file=sys.stderr, flush=True)
            # Group words into ~6s chunks
            chunk, chunk_start = [], None
            for w in word_stamps:
                ws, we = float(w["start"]), float(w["end"])
                if chunk_start is None:
                    chunk_start = ws
                chunk.append({"word": w["word"], "start": ws, "end": we})
                if we - chunk_start >= 6.0:
                    segments.append({
                        "start": chunk_start,
                        "end":   we,
                        "text":  " ".join(x["word"] for x in chunk),
                        "words": chunk,
                    })
                    chunk, chunk_start = [], None
            if chunk:
                segments.append({
                    "start": chunk_start,
                    "end":   chunk[-1]["end"],
                    "text":  " ".join(x["word"] for x in chunk),
                    "words": chunk,
                })

        # --- Fallback: no timestamps at all — single segment from full text ---
        if not segments:
            raw_text = getattr(result, 'text', '') or ''
            if raw_text.strip():
                print("[parakeet] WARNING: no timestamps at all — producing single segment with no word times.",
                      file=sys.stderr, flush=True)
                # Probe audio duration so we at least have a valid end time
                try:
                    dur_probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
                        capture_output=True, text=True, check=True,
                    )
                    duration = float(dur_probe.stdout.strip())
                except Exception:
                    duration = 0.0
                segments.append({
                    "start": 0.0,
                    "end":   duration,
                    "text":  raw_text.strip(),
                    "words": [],
                })

        whisperx_json = {"segments": segments}

        # --- 5. Write outputs ---
        base      = os.path.splitext(os.path.basename(args.audio))[0]
        json_path = os.path.join(args.output_dir, f"{base}.json")
        srt_path  = os.path.join(args.output_dir, f"{base}.srt")

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(whisperx_json, fh, indent=2, ensure_ascii=False)

        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(_build_srt(segments))

        print(f"[parakeet] JSON → {json_path}")
        print(f"[parakeet] SRT  → {srt_path}")

    finally:
        if cleanup_wav:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
