#!/usr/bin/env bash
# Exercise the same Cordis path against an already-running vLLM endpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8800/v1}"
MODEL="${SAH_CORDIS_MODEL:-qwen3.5-9b}"

curl -fsS --max-time 10 "${BASE_URL%/}/models" >/dev/null
OUTPUT="$(OPENAI_BASE_URL="$BASE_URL" SAH_CORDIS_MODEL="$MODEL" \
  "$SCRIPT_DIR/run.sh" 'Reply with one short sentence confirming the harness is operational.')"
[ -n "${OUTPUT//[[:space:]]/}" ] || {
  echo "real model returned no assistant text" >&2
  exit 1
}
printf 'CORDIS_REAL_MODEL_OK model=%s output=%s\n' "$MODEL" "$OUTPUT"
