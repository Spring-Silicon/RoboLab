#!/usr/bin/env bash
set -euo pipefail

APP="$HOME/.local/share/robolab"
BIN="$HOME/.local/bin"
DESKTOP="$HOME/Desktop"
mkdir -p "$APP" "$BIN" "$DESKTOP"
systemctl --user stop spring-openpi-compiled@pi05_compiled_regular.service \
  spring-openpi-compiled@pi05_compiled_optimized.service 2>/dev/null || true

[[ ! -d "$DESKTOP/RoboLab" || -e "$APP/RoboLab" ]] || mv "$DESKTOP/RoboLab" "$APP/RoboLab"
[[ ! -d "$DESKTOP/openpi" || -e "$APP/openpi" ]] || mv "$DESKTOP/openpi" "$APP/openpi"
[[ ! -d "$DESKTOP/robolab-run" || -e "$APP/runner" ]] || mv "$DESKTOP/robolab-run" "$APP/runner"
mkdir -p "$APP/runner/logs"
cp -a "$APP/RoboLab/deploy/cloud/." "$APP/runner/"
chmod +x "$APP/runner"/*.sh "$APP/runner/robolab-run" "$APP/runner/robolab-dashboard"
ln -sfn "$APP/runner/robolab-run" "$BIN/robolab-run"
ln -sfn "$APP/runner/robolab-dashboard" "$BIN/robolab-dashboard"
sudo ln -sfn "$BIN/robolab-run" /usr/local/bin/robolab-run
sudo ln -sfn "$BIN/robolab-dashboard" /usr/local/bin/robolab-dashboard
uv pip install --python "$HOME/.venv-compiled-policy/bin/python" --upgrade \
  -e "$APP/openpi/packages/openpi-client[compiled]"

mkdir -p "$HOME/.config/systemd/user"
cp "$APP/openpi/deploy/user/spring-openpi-compiled@.service" "$HOME/.config/systemd/user/"
cp "$APP/openpi/deploy/robolab-cloud-policy-tunnel.service" "$HOME/.config/systemd/user/"
cp "$APP/openpi/deploy/robolab-cloud-dashboard-tunnel.service" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable robolab-cloud-dashboard-tunnel.service
if [[ -f "$HOME/.config/spring-openpi/cloud.env" ]]; then
  source "$HOME/.config/spring-openpi/cloud.env"
  "$APP/runner/setup-ec2.sh" "$CLOUD_HOST"
fi

cat > "$DESKTOP/RoboLab Dashboard.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=RoboLab Dashboard
Comment=Open RoboLab results
Exec=$BIN/robolab-dashboard
Icon=web-browser
Terminal=false
EOF
cat > "$DESKTOP/Run RoboLab.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Run RoboLab
Comment=Run a RoboLab episode
Exec=gnome-terminal -- $APP/runner/launch-robolab-terminal.sh
Icon=utilities-terminal
Terminal=false
EOF
chmod +x "$DESKTOP/RoboLab Dashboard.desktop" "$DESKTOP/Run RoboLab.desktop"
gio set "$DESKTOP/RoboLab Dashboard.desktop" metadata::trusted true 2>/dev/null || true
gio set "$DESKTOP/Run RoboLab.desktop" metadata::trusted true 2>/dev/null || true
printf 'Installed: robolab-run --help\nDashboard icon: %s\n' "$DESKTOP/RoboLab Dashboard.desktop"
