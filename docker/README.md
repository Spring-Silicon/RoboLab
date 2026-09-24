# Docker

## Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-docker2`)
- Access to `nvcr.io/nvidia/isaac-lab:2.2.0` (base image; use `:2.3.0` for the IsaacSim 5.1 / IsaacLab 2.3 stack)
- To push the built image, a container registry of your own (set `ROBOLAB_REGISTRY` to its image path prefix)

## Build

```bash
# Uses git short SHA as tag by default
./docker/build_docker.sh

# Custom tag
./docker/build_docker.sh my-tag

# Build and push to registry
./docker/build_docker.sh --push

# Custom tag and push
./docker/build_docker.sh my-tag --push
```

The build context is the repo root. A `.dockerignore` at the repo root excludes
`.git/`, build artifacts, and non-essential directories to keep the context small.
Code and `assets/` are baked into the image. Each `assets/` subdirectory is a
separate layer for better caching — code changes don't invalidate asset layers.

## Run

```bash
# Interactive shell with display/GUI forwarding (X11, cache + repo mounts)
./docker/run_docker.sh

# Or specify a custom tag
./docker/run_docker.sh my-tag

```

### Running a single command

```bash
docker run --rm \
    --gpus all \
    --network=host \
    --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
    -e ACCEPT_EULA=Y \
    robolab:<tag> \
    <script.py> [args...]
```

### Running with display (for GUI/rendering)

```bash
docker run --rm -it \
    --gpus all \
    --network=host \
    --entrypoint /bin/bash \
    -e ACCEPT_EULA=Y \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    robolab:<tag>
```

## What's in the image

- **Base**: `nvcr.io/nvidia/isaac-lab:2.2.0` (IsaacSim 5.0) or `:2.3.0` (IsaacSim 5.1), selected via the `ISAACLAB_TAG` build arg (`build_docker.sh --isaac51`)
- **Code**: `robolab/`, `scripts/`, `examples/`, `tests/`
- **Assets**: `assets/` (~6.5GB)
- **Python packages**: Everything in `requirements.txt`, installed via `pip install -e .`
- **System tools**: `htop`, `nvtop`, `tmux`, `vim`, `git-lfs`, `zip`

## Cloud evaluation

Recommended host: one NVIDIA L40S 48 GB (`g6e.2xlarge` or larger), Ubuntu
22.04, 64 GB system RAM, and 500 GB SSD. NVIDIA driver 595.91.07 crashes
Isaac Sim 5.0 during RTX scene initialization on the tested AWS image; use a
compatible production driver such as 580.178.04 or the validated 570.211.01.
An L4 24 GB (`g6.4xlarge`) is sufficient for a one-environment integration
smoke but has less benchmark parallelism headroom.

`cloud-compose.yaml` runs headless Isaac Lab evaluations against a remote
compiled-policy server and serves the results dashboard on port 8080. The
policy host must be reachable from the Docker host, typically over Tailscale.

```bash
# Build the pinned Isaac Lab 2.2 / Isaac Sim 5.0 image.
docker compose -f docker/cloud-compose.yaml build eval

# Run one task against the regular compiled policy.
POLICY_HOST=100.123.6.81 \
POLICY_VARIANT=pi05_compiled_regular \
TASKS="BananaInBowlTask" \
docker compose -f docker/cloud-compose.yaml run --rm eval

# Run the same task against the optimized policy after switching the server.
POLICY_HOST=100.123.6.81 \
POLICY_VARIANT=pi05_compiled_optimized \
TASKS="BananaInBowlTask" \
docker compose -f docker/cloud-compose.yaml run --rm eval

# Browse completed runs at http://<host>:8080.
docker compose -f docker/cloud-compose.yaml up -d dashboard
```

The dashboard is a results viewer; it does not start Isaac Sim evaluations.
Use the `eval` service for launches. Only one compiled policy server may own
the Intel B580 at a time.
