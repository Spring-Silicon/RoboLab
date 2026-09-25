# RoboLab runner

Use an Ubuntu 22.04 NVIDIA GPU EC2 instance with at least 16 GB VRAM, 32 GB RAM, 500 GB disk, and driver 570/580 (not 595). Its security group must allow SSH from Seattle's current public IP; get it on Seattle with `curl -4 https://checkip.amazonaws.com`.

```bash
curl -fsSLO https://raw.githubusercontent.com/Spring-Silicon/RoboLab/main/deploy/cloud/bootstrap-ec2.sh
bash bootstrap-ec2.sh
```

Then on Seattle:

```bash
cd ~/Desktop/robolab-run
./setup-ec2.sh EC2_ADDRESS
./run-episode.sh EC2_ADDRESS optimized AnimalsInBinTask
# model may be: regular or optimized
```

The runner derives Seattle's Tailscale address automatically, switches the policy, records video, restores the regular policy, and logs commands under `logs/`. Run `tailscale ip -4`, browse to `http://<that-IP>:8080`, then select Results → run → task → episode.
