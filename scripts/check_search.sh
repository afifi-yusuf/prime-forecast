#!/usr/bin/env bash
# Search-health preflight: asserts RESULT COUNTS, not HTTP status.
# (Engines that ban an IP return HTTP 200 with an empty result list.)
set -a; source "$(dirname "$0")/../secrets.env" 2>/dev/null; source "$(dirname "$0")/../eval-searxng.env" 2>/dev/null; set +a
N=$(curl -s --max-time 20 "$SEARXNG_URL/search?q=federal+reserve+interest+rate+decision&format=json" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('results',[])))" 2>/dev/null || echo 0)
echo "searxng results: $N"
[ "${N:-0}" -ge 3 ] && echo "SEARCH OK" || { echo "SEARCH DEGRADED — check engine bans (docker logs searxng) or rotate IP"; exit 1; }
