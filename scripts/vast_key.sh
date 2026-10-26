#!/usr/bin/env bash
# Store a Vast.ai API key where the vastai CLI reads it (~/.config/vastai/vast_api_key), then check it.
# Usage: scripts/vast_key.sh   (prompts silently; the key never enters shell history or the repo)
set -euo pipefail
f="${XDG_CONFIG_HOME:-$HOME/.config}/vastai/vast_api_key"
read -rsp "Vast.ai API key: " key; echo
[ -n "$key" ] || { echo "empty key" >&2; exit 1; }
mkdir -p "$(dirname "$f")"
(umask 077; printf '%s' "$key" > "$f")
# curl reads the header from stdin so the key isn't visible in `ps`
if printf 'header = "Authorization: Bearer %s"\n' "$key" |
   curl -fsS -K - https://console.vast.ai/api/v0/users/current/ >/dev/null; then
  echo "ok: key saved to $f and accepted by Vast.ai"
else
  echo "saved to $f but Vast.ai rejected it; check the key" >&2; exit 1
fi
