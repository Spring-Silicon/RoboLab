# RoboLab runner

## EC2

Preconfigured host: `ubuntu@98.80.121.157` (`i-0dd4135e03e6cffdf`, `g6.4xlarge`, NVIDIA L4 24 GB, Ubuntu 22.04, driver 580, 500 GB disk). It is currently running with RoboLab installed.

To create another, use Ubuntu 22.04, an NVIDIA GPU with at least 16 GB VRAM, 32 GB RAM, 500 GB disk, and driver 570/580—not 595. Allow SSH from Seattle's current public IP (`curl -4 https://checkip.amazonaws.com`), then run:

```bash
curl -fsSLO https://raw.githubusercontent.com/Spring-Silicon/RoboLab/main/deploy/cloud/bootstrap-ec2.sh
bash bootstrap-ec2.sh
```

On Seattle:

```bash
cd ~/Desktop/robolab-run
./setup-ec2.sh EC2_ADDRESS
./run-episode.sh EC2_ADDRESS optimized AnimalsInBinTask
# model: regular or optimized
```

The runner switches policies, records video, restores the regular policy, and logs commands under `logs/`. Run `ip -4 route get 1.1.1.1` on Seattle and open `http://<src-IP>:8080` from the same LAN.

## Other providers

Any SSH-reachable x86_64 Ubuntu 22.04 host works if it has an NVIDIA GPU with RT cores, driver 570/580, Docker access, 32 GB RAM, 500 GB disk, outbound internet, and a normal sudo-enabled user. Run the same bootstrap command, then pass its address and SSH user to `run-episode.sh`.
