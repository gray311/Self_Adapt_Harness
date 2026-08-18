#!/usr/bin/env bash
# Run the official DSH headless profile over SAH's OpenAI-compatible endpoint.
set -euo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLATFORM_ROOT="$($SCRIPT_DIR/bootstrap.sh)"
NODE_ROOT="$PLATFORM_ROOT/node-v22.19.0-linux-$(case "$(uname -m)" in x86_64|amd64) printf x64 ;; aarch64|arm64) printf arm64 ;; esac)"
DSH_BIN="$PLATFORM_ROOT/app/node_modules/.bin/dsh"

export PATH="$NODE_ROOT/bin:$PATH"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8800/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export SAH_CORDIS_MODEL="${SAH_CORDIS_MODEL:-qwen3.5-9b}"
export SAH_CORDIS_CONTEXT_WINDOW="${SAH_CORDIS_CONTEXT_WINDOW:-131072}"
export SAH_CORDIS_MAX_TOKENS="${SAH_CORDIS_MAX_TOKENS:-8192}"
export SAH_CORDIS_MAX_RETRIES="${SAH_CORDIS_MAX_RETRIES:-2}"
export DSH_HOME="${DSH_HOME:-$REPO_ROOT/.runtime/cordis/dsh-home}"
export SAH_CORDIS_TRAJECTORY_ROOT="${SAH_CORDIS_TRAJECTORY_ROOT:-$DSH_HOME/trajectories}"
export DSH_TELEMETRY_DISABLED="${DSH_TELEMETRY_DISABLED:-1}"

mkdir -p "$DSH_HOME" "$SAH_CORDIS_TRAJECTORY_ROOT"
# Loader patch rows join the profile's existing Include tree, so relative
# plugin specifiers resolve from the profile root rather than from the patch
# file.  Snapshot the trusted bridge there for every isolated DSH_HOME.
PROFILE_PLUGINS="$DSH_HOME/profiles/headless/plugins"
mkdir -p "$PROFILE_PLUGINS"
cp "$SCRIPT_DIR/plugins/sah-bridge.mjs" "$PROFILE_PLUGINS/sah-bridge.mjs"
WORKSPACE="${SAH_CORDIS_WORKSPACE:-$REPO_ROOT}"
[ -d "$WORKSPACE" ] || {
  echo "Cordis workspace does not exist: $WORKSPACE" >&2
  exit 2
}
cd "$WORKSPACE"
exec "$DSH_BIN" --profile headless --patch "$SCRIPT_DIR/cordis.patch.yml" "$@"
