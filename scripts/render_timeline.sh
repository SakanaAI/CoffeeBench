#!/usr/bin/env bash
# Render the camera-ready overview figure (paper/figures_src/timeline.html) with headless Chrome.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python paper/figures_src/make_timeline.py
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SRC="file://$PWD/paper/figures_src/timeline.html"
OUT=paper/images_cr/timeline
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$OUT.pdf" "$SRC" 2>/dev/null
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size=950,300 --force-device-scale-factor=2 \
  --screenshot="$OUT.png" "$SRC" 2>/dev/null
echo "wrote $OUT.{pdf,png}"
