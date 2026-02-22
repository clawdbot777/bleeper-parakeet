#!/bin/sh
# arr_hook.sh - Radarr/Sonarr custom script hook for Bleeper
#
# Install:
#   1. Place this file somewhere accessible to the arr container
#   2. Radarr/Sonarr → Settings → Connect → Custom Script
#      Script path: /path/to/arr_hook.sh
#      Events: On Import, On Upgrade

BLEEPER_URL="http://bleeper:5000"

# Plex integration — leave empty to skip library refresh
PLEX_URL=""
PLEX_TOKEN=""
PLEX_SECTION_ID="1,2"

# ── Detect arr type and get file path ────────────────────────────────────────
if [ -n "$radarr_eventtype" ]; then
    EVENT="$radarr_eventtype"
    FILE_PATH="$radarr_moviefile_path"
elif [ -n "$sonarr_eventtype" ]; then
    EVENT="$sonarr_eventtype"
    FILE_PATH="$sonarr_episodefile_path"
else
    echo "[bleeper] No event detected - skipping"
    exit 0
fi

case "$EVENT" in
    Download|EpisodeFileImport|MovieFileImport) ;;
    Test) echo "[bleeper] Test event received - OK"; exit 0 ;;
    *) echo "[bleeper] Skipping event: $EVENT"; exit 0 ;;
esac

if [ -z "$FILE_PATH" ]; then
    echo "[bleeper] No file path in event - skipping"
    exit 0
fi

echo "[bleeper] Submitting: $FILE_PATH"

# ── Build JSON payload ────────────────────────────────────────────────────────
PAYLOAD=$(cat <<EOF
{
  "filename": "$FILE_PATH",
  "plex_url": "$PLEX_URL",
  "plex_token": "$PLEX_TOKEN",
  "plex_section_id": "$PLEX_SECTION_ID"
}
EOF
)

# ── POST to bleeper API ───────────────────────────────────────────────────────
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "$BLEEPER_URL/api/process_full" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    --connect-timeout 10 \
    --max-time 30)

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -1)

if [ "$HTTP_CODE" = "202" ]; then
    JOB_ID=$(echo "$BODY" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
    POSITION=$(echo "$BODY" | grep -o '"position":[0-9]*' | cut -d':' -f2)
    echo "[bleeper] Queued OK - job_id: $JOB_ID (position: $POSITION)"
    exit 0
else
    echo "[bleeper] Failed - HTTP $HTTP_CODE: $BODY"
    exit 1
fi
