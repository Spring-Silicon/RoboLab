# RoboLab runner

## EC2

We have a preconfigured host for you at `ubuntu@98.80.121.157` (it is a g6.4xlarge with this setup: L4 24 GB, Ubuntu 22.04, driver 580, 500 GB disk). It is currently already running with RoboLab installed.

To create another, use Ubuntu 22.04, an NVIDIA GPU with at least 16 GB VRAM, 32 GB RAM, 500 GB disk, and driver 570/580. Notably, not 595.

Allow SSH from Cleveland's current public IP (`curl -4 https://checkip.amazonaws.com`), then run:

```bash
curl -fsSLO https://raw.githubusercontent.com/Spring-Silicon/RoboLab/main/deploy/cloud/bootstrap-ec2.sh
bash bootstrap-ec2.sh
```

On Cleveland:

```bash
cd ~/Desktop/robolab-run
./setup-ec2.sh EC2_ADDRESS
./run-episode.sh EC2_ADDRESS pi05_spring_optimized AnimalsInBinTask
# model: pi0, pi0_fast, pi05, paligemma, paligemma_fast, pi05_spring_regular, or pi05_spring_optimized
```


## Other providers

Any SSHable x86_64 Ubuntu 22.04 host will also work if it has an NVIDIA GPU with RT cores, driver 570/580, Docker access, 32 GB RAM, 500 GB disk, outbound internet, and a sudo-enabled user. Run the same bootstrap command and then pass its address and SSH user to `run-episode.sh`.
