# RoboLab runner

Use an Ubuntu 22.04 NVIDIA GPU EC2 instance with at least 16 GB VRAM, 32 GB RAM, 500 GB disk, a working NVIDIA driver (570/580; not 595), and inbound SSH from Seattle (`208.64.29.18`). On the EC2 instance run:

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

The runner switches the Seattle policy, opens the SSH tunnel, runs one video-recorded episode, restores the regular policy, and logs every command under `logs/`. Results appear at **http://100.123.6.81:8080**. Select Results → the output run → task → episode to play `viewport` or `recording`; the EC2 dashboard is also at `http://EC2_ADDRESS:8080` if its security group allows that port.
