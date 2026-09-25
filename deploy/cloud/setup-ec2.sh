#!/usr/bin/env bash
set -euo pipefail

HOST="${1:?usage: setup-ec2.sh <ec2-address> [ssh-user]}"
USER_NAME="${2:-ubuntu}"
KEY="$HOME/.ssh/robolab-cloud-tunnel"
DASHBOARD_IP="${ROBOLAB_DASHBOARD_IP:-$(ip -4 route get 1.1.1.1 | sed -n 's/.* src \([^ ]*\).*/\1/p')}"
[[ "$HOST" =~ ^[A-Za-z0-9.:-]+$ && "$DASHBOARD_IP" =~ ^[0-9.]+$ ]] \
  || { echo "invalid GPU host or runner LAN address" >&2; exit 2; }

ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  "$USER_NAME@$HOST" 'curl -fsS http://127.0.0.1:8080/api/runs >/dev/null'
mkdir -p "$HOME/.config/spring-openpi"
printf 'CLOUD_HOST=%s\nDASHBOARD_FORWARD=%s:8080:127.0.0.1:8080\n' "$HOST" "$DASHBOARD_IP" \
  > "$HOME/.config/spring-openpi/cloud.env"
systemctl --user restart robolab-cloud-policy-tunnel.service
for _ in $(seq 1 20); do
  curl -fsS "http://$DASHBOARD_IP:8080/api/runs" >/dev/null && break
  sleep 1
done
curl -fsS "http://$DASHBOARD_IP:8080/api/runs" >/dev/null
printf 'READY EC2=%s UI=http://%s:8080\n' "$HOST" "$DASHBOARD_IP"
