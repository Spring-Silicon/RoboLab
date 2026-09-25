#!/usr/bin/env bash
set -euo pipefail

HOST="${1:?usage: setup-ec2.sh <ec2-address> [ssh-user]}"
USER_NAME="${2:-ubuntu}"
KEY="$HOME/.ssh/robolab-cloud-tunnel"
[[ "$HOST" =~ ^[A-Za-z0-9.:-]+$ ]] || { echo "invalid EC2 address" >&2; exit 2; }

ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  "$USER_NAME@$HOST" 'curl -fsS http://127.0.0.1:8080/api/runs >/dev/null'
mkdir -p "$HOME/.config/spring-openpi"
printf 'CLOUD_HOST=%s\nDASHBOARD_FORWARD=100.123.6.81:8080:127.0.0.1:8080\n' "$HOST" \
  > "$HOME/.config/spring-openpi/cloud.env"
systemctl --user restart robolab-cloud-policy-tunnel.service
for _ in $(seq 1 20); do
  curl -fsS http://100.123.6.81:8080/api/runs >/dev/null && break
  sleep 1
done
curl -fsS http://100.123.6.81:8080/api/runs >/dev/null
printf 'READY EC2=%s UI=http://100.123.6.81:8080\n' "$HOST"
