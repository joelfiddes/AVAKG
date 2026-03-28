#!/bin/bash
# Deploy AvaFrame WebUI on myserver
# Run this ON the server after git pull
set -e

echo "=== AvaFrame WebUI Deployment ==="

INSTALL_DIR="/opt/avaframe-webui"
VENV_DIR="$INSTALL_DIR/venv"
APP_DIR="$INSTALL_DIR/app"
DATA_DIR="/var/lib/avaframe"
SERVICE_NAME="avaframe-webui"

# 1. Create directories
echo "[1/5] Creating directories..."
sudo mkdir -p "$INSTALL_DIR" "$DATA_DIR"
sudo chown joel:joel "$INSTALL_DIR" "$DATA_DIR"

# 2. Copy app files
echo "[2/5] Copying application files..."
mkdir -p "$APP_DIR"
cp -r "$(dirname "$0")"/* "$APP_DIR/"

# 3. Create Python venv and install dependencies
echo "[3/5] Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet flask pyyaml avaframe rasterio geopandas fiona shapely scipy \
    pillow matplotlib pandas requests dem-stitcher pyproj

# 4. Create systemd service
echo "[4/5] Installing systemd service..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << 'SERVICEEOF'
[Unit]
Description=AvaFrame WebUI
After=network.target

[Service]
Type=simple
User=joel
Group=joel
WorkingDirectory=/opt/avaframe-webui/app
ExecStart=/opt/avaframe-webui/venv/bin/python app.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=AVAFRAME_DATA_DIR=/var/lib/avaframe
Environment=AVAFRAME_URL_PREFIX=/avaframe

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl restart $SERVICE_NAME

# 5. Configure Caddy reverse proxy
echo "[5/5] Configuring Caddy..."
CADDY_SNIPPET='
    # AvaFrame WebUI
    handle_path /avaframe/* {
        reverse_proxy localhost:5050
    }
    handle /avaframe {
        redir /avaframe/ permanent
    }
'

CADDYFILE="/etc/caddy/Caddyfile"
if ! grep -q "avaframe" "$CADDYFILE" 2>/dev/null; then
    echo ""
    echo "  Add this INSIDE the apps.mountainfutures.ch block in $CADDYFILE:"
    echo "$CADDY_SNIPPET"
    echo "  Then run: sudo systemctl reload caddy"
else
    echo "  Caddy config already contains avaframe entry"
fi

echo ""
echo "=== Deployment complete ==="
echo "  Service: sudo systemctl status $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo "  URL:     https://apps.mountainfutures.ch/avaframe/"
echo ""
