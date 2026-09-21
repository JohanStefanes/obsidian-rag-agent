#!/bin/bash
# Build VaultRAG.app: a macOS bundle that starts the local server and opens the
# UI in a native window. The launcher embeds absolute paths (this machine), so
# it works when double-clicked from Finder with a minimal environment.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$REPO/packaging/build"
APP="${1:-$REPO/dist/VaultRAG.app}"
UV="$(command -v uv || echo /opt/homebrew/bin/uv)"
OLLAMA="$(command -v ollama || echo /opt/homebrew/opt/ollama/bin/ollama)"

echo "Repo:   $REPO"
echo "App:    $APP"
echo "uv:     $UV"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$BUILD"

# --- icon ---------------------------------------------------------------
echo "Generating icon ..."
"$UV" run --with pillow python "$REPO/packaging/make_icon.py" "$BUILD/icon.png"
ICONSET="$BUILD/VaultRAG.iconset"
rm -rf "$ICONSET"; mkdir -p "$ICONSET"
for s in 16 32 64 128 256 512 1024; do
  sips -z $s $s "$BUILD/icon.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
done
# Retina (@2x) variants expected by iconutil.
cp "$ICONSET/icon_32x32.png"   "$ICONSET/icon_16x16@2x.png"
cp "$ICONSET/icon_64x64.png"   "$ICONSET/icon_32x32@2x.png"
cp "$ICONSET/icon_256x256.png" "$ICONSET/icon_128x128@2x.png"
cp "$ICONSET/icon_512x512.png" "$ICONSET/icon_256x256@2x.png"
cp "$ICONSET/icon_1024x1024.png" "$ICONSET/icon_512x512@2x.png"
rm -f "$ICONSET/icon_64x64.png" "$ICONSET/icon_1024x1024.png"
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/VaultRAG.icns"

# --- Info.plist ---------------------------------------------------------
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>VaultRAG</string>
  <key>CFBundleDisplayName</key>       <string>VaultRAG</string>
  <key>CFBundleIdentifier</key>        <string>dev.johan.vaultrag</string>
  <key>CFBundleVersion</key>           <string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleExecutable</key>        <string>VaultRAG</string>
  <key>CFBundleIconFile</key>          <string>VaultRAG</string>
  <key>NSHighResolutionCapable</key>   <true/>
  <key>LSMinimumSystemVersion</key>    <string>12.0</string>
</dict>
</plist>
PLIST

# --- launcher -----------------------------------------------------------
LAUNCHER="$APP/Contents/MacOS/VaultRAG"
cat > "$LAUNCHER" <<LAUNCH
#!/bin/bash
# Double-click launcher. Absolute paths only (Finder gives a minimal PATH).
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
REPO="$REPO"
UV="$UV"
OLLAMA="$OLLAMA"

# Ensure Ollama is running (needed for embeddings + chat).
if ! curl -s -m 1 http://localhost:11434/api/version >/dev/null 2>&1; then
  ( OLLAMA_FLASH_ATTENTION=1 "\$OLLAMA" serve >/tmp/vaultrag-ollama.log 2>&1 & )
  for i in \$(seq 1 30); do
    curl -s -m 1 http://localhost:11434/api/version >/dev/null 2>&1 && break
    sleep 0.3
  done
fi

cd "\$REPO"
exec "\$UV" run --extra web --extra app vaultrag app >/tmp/vaultrag-app.log 2>&1
LAUNCH
chmod +x "$LAUNCHER"

# Refresh Finder's icon cache for the new bundle.
touch "$APP"

echo "Built $APP"
