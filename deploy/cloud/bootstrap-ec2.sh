#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ROBOLAB_IMAGE:-public.ecr.aws/m4l3e1i0/spring-silicon/robolab:2026-09-25}"
SEATTLE_KEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAAzmkK6h/tPp7Vx1GbRkMcgy6BEN7ItDImH4HpYyknv robolab-cloud-tunnel'

command -v nvidia-smi >/dev/null || { echo "Use an Ubuntu 22.04 NVIDIA GPU AMI with drivers installed." >&2; exit 2; }
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1 | head -1)"
[[ "$DRIVER" != 595 ]] || { echo "NVIDIA 595 crashes Isaac Sim 5.0; install driver 580 and reboot, then rerun." >&2; exit 2; }

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl gnupg
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh
fi
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
sudo usermod -aG docker "$USER"
mkdir -p "$HOME/.ssh" "$HOME/robolab-output" "$HOME/.config/robolab-dashboard"
chmod 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
grep -qxF "$SEATTLE_KEY" "$HOME/.ssh/authorized_keys" || printf '%s\n' "$SEATTLE_KEY" >> "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"

sudo docker pull "$IMAGE"
sudo docker rm -f robolab-dashboard docker-dashboard-1 >/dev/null 2>&1 || true
sudo docker run -d --name robolab-dashboard --restart unless-stopped --network host \
  -v "$HOME/robolab-output:/workspace/robolab/output:ro" \
  -v "$HOME/.config/robolab-dashboard:/root/.config/robolab-dashboard" \
  --entrypoint /workspace/isaaclab/_isaac_sim/python.sh "$IMAGE" \
  -m dashboard.cli --output-dir /workspace/robolab/output --host 0.0.0.0 --port 8080 >/dev/null

sudo docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE" --query-gpu=name,memory.total --format=csv,noheader
printf 'READY image=%s dashboard=http://%s:8080\n' "$IMAGE" "$(hostname -I | awk '{print $1}')"
