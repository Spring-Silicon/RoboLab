#!/usr/bin/env bash
set -euo pipefail

HOST="${1:?usage: setup-ec2.sh <ec2-address> [ssh-user]}"
USER_NAME="${2:-ubuntu}"
KEY="$HOME/.ssh/robolab-cloud-tunnel"
TAIL_IP="${SEATTLE_TAILSCALE_IP:-$(tailscale ip -4)}"
[[ "$HOST" =~ ^[A-Za-z0-9.:-]+$ && "$TAIL_IP" =~ ^[0-9a-fA-F:.]+$ ]] \
  || { echo "invalid EC2 or Tailscale address" >&2; exit 2; }

ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  "$USER_NAME@$HOST" 'curl -fsS http://127.0.0.1:8080/api/runs >/dev/null'
mkdir -p "$HOME/.config/spring-openpi"
printf 'CLOUD_HOST=%s\nDASHBOARD_FORWARD=%s:8080:127.0.0.1:8080\n' "$HOST" "$TAIL_IP" \
  > "$HOME/.config/spring-openpi/cloud.env"
systemctl --user restart robolab-cloud-policy-tunnel.service
for _ in $(seq 1 20); do
  curl -fsS "http://$TAIL_IP:8080/api/runs" >/dev/null && break
  sleep 1
done
curl -fsS "http://$TAIL_IP:8080/api/runs" >/dev/null
printf 'READY EC2=%s UI=http://%s:8080\n' "$HOST" "$TAIL_IP"
