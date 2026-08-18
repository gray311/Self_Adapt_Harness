#!/usr/bin/env bash
# Install a pinned Node runtime and the pinned DSH/Cordis dependency closure.
# The installation is architecture-scoped so the x86 login node and aarch64
# GB200 workers can safely share this repository.
set -euo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODE_VERSION="22.19.0"
PLATFORM="linux"

case "$(uname -m)" in
  x86_64|amd64)
    NODE_ARCH="x64"
    NODE_SHA256="c0649af18e6a24f6fe5535a3e86b341dd49a8e71117c8b68bde973ef834f16f2"
    ;;
  aarch64|arm64)
    NODE_ARCH="arm64"
    NODE_SHA256="0b2d9f564b6594222a62c82e1df2efe119dd4a4aff29644f4dd325bf360b6bcc"
    ;;
  *)
    echo "unsupported architecture for the pinned Cordis runtime: $(uname -m)" >&2
    exit 2
    ;;
esac

RUNTIME_BASE="${SAH_CORDIS_RUNTIME_DIR:-$REPO_ROOT/.runtime/cordis}"
PLATFORM_ROOT="$RUNTIME_BASE/$PLATFORM-$NODE_ARCH"
NODE_ROOT="$PLATFORM_ROOT/node-v$NODE_VERSION-$PLATFORM-$NODE_ARCH"
APP_ROOT="$PLATFORM_ROOT/app"
DOWNLOAD_DIR="$RUNTIME_BASE/downloads"
ARCHIVE_NAME="node-v$NODE_VERSION-$PLATFORM-$NODE_ARCH.tar.xz"
ARCHIVE="$DOWNLOAD_DIR/$ARCHIVE_NAME"
NODE_URL="https://nodejs.org/dist/v$NODE_VERSION/$ARCHIVE_NAME"

mkdir -p "$DOWNLOAD_DIR" "$PLATFORM_ROOT"

valid_archive() {
  [ -f "$ARCHIVE" ] && printf '%s  %s\n' "$NODE_SHA256" "$ARCHIVE" | sha256sum --check --status
}

if [ ! -x "$NODE_ROOT/bin/node" ]; then
  if ! valid_archive; then
    if [ -e "$ARCHIVE" ]; then
      echo "cached Node archive has the wrong checksum: $ARCHIVE" >&2
      exit 1
    fi
    echo "[cordis] downloading Node.js $NODE_VERSION for $PLATFORM-$NODE_ARCH" >&2
    curl -fL "$NODE_URL" -o "$ARCHIVE.partial"
    printf '%s  %s\n' "$NODE_SHA256" "$ARCHIVE.partial" | sha256sum --check --status
    mv "$ARCHIVE.partial" "$ARCHIVE"
  fi

  EXTRACT_DIR="$(mktemp -d "$PLATFORM_ROOT/node-extract.XXXXXX")"
  cleanup_extract() { rm -rf -- "$EXTRACT_DIR"; }
  trap cleanup_extract EXIT
  tar -xJf "$ARCHIVE" -C "$EXTRACT_DIR"
  mv "$EXTRACT_DIR/node-v$NODE_VERSION-$PLATFORM-$NODE_ARCH" "$NODE_ROOT"
  trap - EXIT
  cleanup_extract
fi

ACTUAL_VERSION="$($NODE_ROOT/bin/node --version)"
if [ "$ACTUAL_VERSION" != "v$NODE_VERSION" ]; then
  echo "unexpected Node version at $NODE_ROOT: $ACTUAL_VERSION" >&2
  exit 1
fi

LOCK_SHA256="$(sha256sum "$SCRIPT_DIR/package-lock.json" | awk '{print $1}')"
STAMP="$APP_ROOT/.sah-cordis-lock.sha256"
INSTALLED_SHA256=""
if [ -f "$STAMP" ]; then
  INSTALLED_SHA256="$(sed -n '1p' "$STAMP")"
fi

if [ ! -x "$APP_ROOT/node_modules/.bin/dsh" ] || [ "$INSTALLED_SHA256" != "$LOCK_SHA256" ]; then
  echo "[cordis] installing pinned DSH/Cordis packages for $PLATFORM-$NODE_ARCH" >&2
  mkdir -p "$APP_ROOT"
  cp "$SCRIPT_DIR/package.json" "$SCRIPT_DIR/package-lock.json" "$APP_ROOT/"
  export PATH="$NODE_ROOT/bin:$PATH"
  export npm_config_cache="$RUNTIME_BASE/npm-cache"
  npm ci --prefix "$APP_ROOT" --no-audit --no-fund
  printf '%s\n' "$LOCK_SHA256" > "$STAMP"
fi

# Machine-readable stdout for run.sh; all progress belongs on stderr.
printf '%s\n' "$PLATFORM_ROOT"
