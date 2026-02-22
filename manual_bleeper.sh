#!/bin/sh
# manual_bleeper.sh - Manually trigger bleeper for a media file
# Usage: ./manual_bleeper.sh /data/media/movies/Icefall\ \(2025\)\ .../file.mkv
# (tab completion works — no quotes needed)

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/file.mkv"
    exit 1
fi

FILE_PATH="$@"

if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File not found: $FILE_PATH"
    exit 1
fi

# Extract movie title from parent directory name
TITLE=$(basename "$(dirname "$FILE_PATH")")

echo "[manual] File:  $FILE_PATH"
echo "[manual] Title: $TITLE"
echo ""

radarr_eventtype=Download \
radarr_moviefile_path="$FILE_PATH" \
radarr_movie_title="$TITLE" \
/data/plugin/arr_hook.sh