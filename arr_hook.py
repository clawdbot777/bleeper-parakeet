#!/usr/bin/env python3
"""
arr_hook.py - Radarr / Sonarr custom script hook for Bleeper automation.

Install:
  1. Copy this file somewhere accessible (e.g. /opt/bleeper/arr_hook.py)
  2. In Radarr/Sonarr → Settings → Connect → add a "Custom Script" connection.
     - Script path: /opt/bleeper/arr_hook.py
     - Events to trigger on: "On Import" (and optionally "On Upgrade")

Environment variables injected by Radarr/Sonarr:
  Radarr:
    radarr_eventtype              e.g. "Download"
    radarr_moviefile_path         full path to the imported file
    radarr_movie_title            movie title

  Sonarr:
    sonarr_eventtype              e.g. "Download"
    sonarr_episodefile_path       full path to the imported file
    sonarr_series_title           series title

Configuration:
  Set the variables in the CONFIG section below, or use environment variables.
"""

import os
import sys
import json
import shutil
import logging
import requests
import argparse

# ---------------------------------------------------------------------------
# CONFIG – edit these or override via environment variables
# ---------------------------------------------------------------------------

BLEEPER_URL     = os.environ.get("BLEEPER_URL",     "http://localhost:5000")
BLEEPER_UPLOAD  = os.environ.get("BLEEPER_UPLOAD",  "/tmp/uploads")

# Plex integration (optional – leave empty to skip)
PLEX_URL        = os.environ.get("PLEX_URL",        "")
PLEX_TOKEN      = os.environ.get("PLEX_TOKEN",      "")
PLEX_SECTION_ID = os.environ.get("PLEX_SECTION_ID", "")

# whisperX overrides (optional – bleeper uses sensible defaults)
WHISPERX_SETTINGS = {
    # "model":        "large-v3",
    # "language":     "en",
    # "batch_size":   20,
    # "compute_type": "float16",
}

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("arr_hook")


def get_event_file() -> tuple[str, str]:
    """
    Detect whether we're running under Radarr or Sonarr and return
    (event_type, file_path).
    """
    # Radarr
    if os.environ.get("radarr_eventtype"):
        event = os.environ["radarr_eventtype"]
        path  = os.environ.get("radarr_moviefile_path", "")
        title = os.environ.get("radarr_movie_title", "unknown")
        logger.info(f"Radarr event: {event} – {title} – {path}")
        return event, path

    # Sonarr
    if os.environ.get("sonarr_eventtype"):
        event = os.environ["sonarr_eventtype"]
        path  = os.environ.get("sonarr_episodefile_path", "")
        title = os.environ.get("sonarr_series_title", "unknown")
        logger.info(f"Sonarr event: {event} – {title} – {path}")
        return event, path

    return "", ""


def submit_to_bleeper(file_path: str) -> str | None:
    """
    Copy the media file to the bleeper upload folder and submit a
    process_full job.  Returns job_id on success, None on failure.
    """
    filename = os.path.basename(file_path)
    dest     = os.path.join(BLEEPER_UPLOAD, filename)

    # Copy only if not already in the upload folder
    if os.path.abspath(file_path) != os.path.abspath(dest):
        logger.info(f"Copying {file_path} → {dest}")
        shutil.copy2(file_path, dest)
    else:
        logger.info(f"File already in upload folder: {dest}")

    payload = {
        "filename":         filename,
        "whisperx_settings": WHISPERX_SETTINGS or None,
    }
    if PLEX_URL and PLEX_TOKEN and PLEX_SECTION_ID:
        payload["plex_url"]        = PLEX_URL
        payload["plex_token"]      = PLEX_TOKEN
        payload["plex_section_id"] = PLEX_SECTION_ID

    try:
        resp = requests.post(
            f"{BLEEPER_URL}/api/process_full",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        job_id = resp.json().get("job_id")
        logger.info(f"Submitted to bleeper – job_id: {job_id}")
        return job_id
    except requests.RequestException as e:
        logger.error(f"Failed to submit to bleeper: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Arr hook: submit imported media files to the Bleeper pipeline."
    )
    parser.add_argument(
        "--file", "-f",
        help="Manually specify a file path (bypasses env-var detection).",
    )
    args = parser.parse_args()

    if args.file:
        event, file_path = "Manual", args.file
    else:
        event, file_path = get_event_file()

    if not file_path:
        logger.info("No file path found – nothing to do (or test event).")
        return 0

    if event not in ("Download", "Manual", "MovieFileImport", "EpisodeFileImport"):
        logger.info(f"Event {event!r} – skipping (not an import event).")
        return 0

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1

    job_id = submit_to_bleeper(file_path)
    return 0 if job_id else 1


if __name__ == "__main__":
    sys.exit(main())
