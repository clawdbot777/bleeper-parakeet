#!/usr/bin/env python3
"""
watcher.py - Drop-folder watcher for Bleeper automation.

Monitors a folder for new video files and automatically submits them
to the Bleeper pipeline.  Useful for manual drops or any workflow
that doesn't go through Radarr/Sonarr.

Usage:
    python watcher.py --watch /path/to/watch/folder

    # Or set via env var:
    WATCH_FOLDER=/media/incoming python watcher.py

Requires: watchdog  (pip install watchdog)
"""

import os
import sys
import time
import logging
import argparse
import requests
import shutil

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
except ImportError:
    print("watchdog not installed. Run: pip install watchdog")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BLEEPER_URL     = os.environ.get("BLEEPER_URL",     "http://localhost:5000")
BLEEPER_UPLOAD  = os.environ.get("BLEEPER_UPLOAD",  "/tmp/uploads")
WATCH_FOLDER    = os.environ.get("WATCH_FOLDER",    "/media/incoming")

PLEX_URL        = os.environ.get("PLEX_URL",        "")
PLEX_TOKEN      = os.environ.get("PLEX_TOKEN",      "")
PLEX_SECTION_ID = os.environ.get("PLEX_SECTION_ID", "")

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts"}

# How long to wait after a file appears before submitting (seconds).
# Gives time for large file writes to complete.
STABLE_WAIT_SECONDS = 30

# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("watcher")


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def _wait_until_stable(path: str, wait: int = STABLE_WAIT_SECONDS,
                       max_retries: int = 60) -> bool:
    """Wait until a file stops growing.  Returns True if stable, False if gone."""
    for attempt in range(max_retries):
        logger.info(f"Waiting {wait}s for file to stabilise: {path} (attempt {attempt + 1})")
        time.sleep(wait)
        if not os.path.exists(path):
            return False
        size_before = os.path.getsize(path)
        time.sleep(5)
        if not os.path.exists(path):
            return False
        size_after = os.path.getsize(path)
        if size_before == size_after:
            return True
        logger.info("File still growing – waiting another cycle.")
    logger.warning(f"File {path} still growing after {max_retries} retries – giving up.")
    return False


def submit_file(file_path: str) -> None:
    filename = os.path.basename(file_path)
    dest     = os.path.join(BLEEPER_UPLOAD, filename)

    if os.path.abspath(file_path) != os.path.abspath(dest):
        logger.info(f"Copying {file_path} → {dest}")
        shutil.copy2(file_path, dest)

    payload: dict = {"filename": filename}
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
        logger.info(f"Submitted {filename} → bleeper job_id={job_id}")
    except requests.RequestException as e:
        logger.error(f"Failed to submit {filename}: {e}")


class VideoHandler(FileSystemEventHandler):
    def _handle(self, path: str) -> None:
        if not _is_video(path):
            return
        import threading
        def _process():
            if _wait_until_stable(path):
                submit_file(path)
        threading.Thread(target=_process, daemon=True).start()

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._handle(event.dest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a folder and auto-submit videos to Bleeper.")
    parser.add_argument("--watch", default=WATCH_FOLDER, help="Folder to watch.")
    args = parser.parse_args()

    watch_folder = args.watch
    if not os.path.isdir(watch_folder):
        os.makedirs(watch_folder, exist_ok=True)
        logger.info(f"Created watch folder: {watch_folder}")

    logger.info(f"Watching: {watch_folder}  →  Bleeper: {BLEEPER_URL}")

    observer = Observer()
    observer.schedule(VideoHandler(), watch_folder, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
