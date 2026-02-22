# Unraid Template

## Quick Import

1. Copy [`bleeper.xml`](bleeper.xml) to your Unraid server:

```bash
# SSH into Unraid, then:
mkdir -p /boot/config/plugins/dockerMan/templates-user
curl -o /boot/config/plugins/dockerMan/templates-user/bleeper.xml \
  https://raw.githubusercontent.com/jakezp/bleep_test/main/unraid/bleeper.xml
```

2. Go to **Docker** tab in Unraid → **Add Container** → the template will appear as "bleeper" in the dropdown.

Or go to **Apps** (Community Applications) → **My Templates** → it should show up there.

---

## GPU passthrough

Install the **Nvidia-Driver** plugin from Community Applications first, then the `--gpus all` extra parameter in the template handles the rest.

If you don't have a GPU, remove `--gpus all` from Extra Parameters and set `WHISPERX_COMPUTE_TYPE` to `int8`.

---

## Arr integration

After the container is running, add the hook to Radarr/Sonarr:

1. Settings → Connect → + → Custom Script
2. Name: `Bleeper`
3. On Import: ✅ On Upgrade: ✅
4. Path: `/mnt/user/appdata/bleeper/arr_hook.py`
5. Set env vars (or edit the CONFIG block in `arr_hook.py`):
   - `BLEEPER_URL=http://[UNRAID-IP]:5000`
   - `BLEEPER_UPLOAD=/mnt/user/appdata/bleeper/uploads`
   - `PLEX_URL`, `PLEX_TOKEN`, `PLEX_SECTION_ID`
