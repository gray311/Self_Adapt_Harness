#!/usr/bin/env bash
# End-to-end test: Cordis tree -> DSH agent loop -> OpenAI adapter -> mock model.
set -euo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_BASE="${SAH_CORDIS_RUNTIME_DIR:-$REPO_ROOT/.runtime/cordis}"
mkdir -p "$RUNTIME_BASE"
SMOKE_DIR="$(mktemp -d "$RUNTIME_BASE/smoke.XXXXXX")"
READY_FILE="$SMOKE_DIR/ready"
REQUEST_FILE="$SMOKE_DIR/requests.jsonl"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf -- "$SMOKE_DIR"
}
trap cleanup EXIT

python3 "$SCRIPT_DIR/mock_openai_server.py" \
  --ready-file "$READY_FILE" --request-file "$REQUEST_FILE" &
SERVER_PID=$!
for _ in $(seq 1 100); do
  [ -s "$READY_FILE" ] && break
  kill -0 "$SERVER_PID" 2>/dev/null || {
    echo "mock model exited before becoming ready" >&2
    exit 1
  }
  sleep 0.1
done
[ -s "$READY_FILE" ] || { echo "mock model readiness timeout" >&2; exit 1; }
PORT="$(sed -n '1p' "$READY_FILE")"

export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export SAH_CORDIS_MODEL="cordis-smoke-model"
export SAH_CORDIS_CONTEXT_WINDOW="4096"
export SAH_CORDIS_MAX_TOKENS="256"
export DSH_HOME="$SMOKE_DIR/dsh-home"
export SAH_CORDIS_TRAJECTORY_ROOT="$DSH_HOME/trajectories"

OUTPUT="$($SCRIPT_DIR/run.sh 'Return the model smoke marker and stop.')"
case "$OUTPUT" in
  *CORDIS_MODEL_OK*) ;;
  *)
    echo "Cordis smoke returned unexpected output: $OUTPUT" >&2
    exit 1
    ;;
esac
python3 "$SCRIPT_DIR/export_trajectory.py" \
  "$SAH_CORDIS_TRAJECTORY_ROOT" "$SMOKE_DIR/trajectory.jsonl" >/dev/null

python3 -c '
import json, pathlib, sys
rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
assert rows, "Cordis did not send a model request"
assert all(row.get("model") == sys.argv[2] for row in rows), rows
assert any(row.get("messages") for row in rows), rows
' "$REQUEST_FILE" "$SAH_CORDIS_MODEL"
python3 -c '
import json, pathlib, sys
rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
types = {row.get("type") for row in rows}
assert {"session", "request/header", "assistant/message", "turn/end"} <= types, types
' "$SMOKE_DIR/trajectory.jsonl"

printf 'CORDIS_SMOKE_OK model_requests=%s output=%s\n' \
  "$(wc -l < "$REQUEST_FILE" | tr -d ' ')" "$OUTPUT"
