#!/bin/bash
# ================================================================
# Google Drive ZIP Downloader → Extracts into current directory
# Usage:
#   ./setup_from_drive.sh
# ================================================================

set -euo pipefail

say()  { echo -e "\033[1;34m$*\033[0m"; }
err()  { echo -e "\033[1;31m$*\033[0m" >&2; }
need() { command -v "$1" >/dev/null 2>&1; }

DATA="$PWD"

# ---- Google Drive link (hardcoded)
GDRIVE_LINK="https://drive.google.com/file/d/1vA0rGsSI73EcIcqDT2qUCyHDgLUsMc89/view?usp=sharing"

# --- ensure unzip and gdown ---
if ! need unzip; then
  say "Installing unzip..."
  if need apt-get; then sudo apt-get update -y && sudo apt-get install -y unzip
  elif need dnf; then sudo dnf install -y unzip
  elif need yum; then sudo yum install -y unzip
  elif need pacman; then sudo pacman -Sy --noconfirm unzip
  elif need zypper; then sudo zypper install -y unzip
  elif need apk; then sudo apk add --no-cache unzip
  else err "Install unzip manually."; exit 1; fi
fi

if ! need gdown; then
  say "Installing gdown..."
  if need pip3; then pip3 install --user gdown
  elif need pip; then pip install --user gdown
  else err "pip not found. Please install Python3 & pip."; exit 1; fi
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- download ZIP ---
OUT_ZIP="$DATA/dataset.zip"
say "⬇️  Downloading ZIP from Google Drive..."
if gdown --fuzzy "$GDRIVE_LINK" -O "$OUT_ZIP"; then :; else
  say "Retrying in legacy mode (--id)..."
  gdown --fuzzy --id "$GDRIVE_LINK" -O "$OUT_ZIP"
fi

[ -f "$OUT_ZIP" ] || { err "❌ Download failed."; exit 1; }

# --- verify ZIP ---
if file "$OUT_ZIP" | grep -qi "zip archive data"; then
  say "📦 Valid ZIP file detected."
else
  err "❌ The downloaded file is not a valid ZIP."
  err "Ensure Google Drive sharing is set to: Anyone with the link → Viewer."
  exit 1
fi

# --- extract ---
say "📂 Extracting ZIP into: $DATA"
unzip -o "$OUT_ZIP" -d "$DATA" >/dev/null

# --- cleanup ---
rm -f "$OUT_ZIP"

say "✅ Done! Files extracted to: $DATA"