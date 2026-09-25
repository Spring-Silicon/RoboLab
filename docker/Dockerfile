# IsaacLab/IsaacSim stack is selected by the base image tag:
#   2.2.0 -> IsaacLab 2.2.0 / IsaacSim 5.0   (default)
#   2.3.0 -> IsaacLab 2.3.x  / IsaacSim 5.1
# The project's isaac extras (isaac50/isaac51) are NOT installed here: the
# base image already provides the matching IsaacLab/IsaacSim, so `pip install
# -e .` only pulls the non-isaac deps.
ARG ISAACLAB_TAG=2.2.0
FROM nvcr.io/nvidia/isaac-lab:${ISAACLAB_TAG}

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    htop \
    iputils-ping \
    git-lfs \
    nvtop \
    tmux \
    vim \
    zip \
    && rm -rf /var/lib/apt/lists/*

# openpi-client pinned to the Spring Silicon compiled-policy gateway commit.
ARG OPENPI_REPO=https://github.com/Spring-Silicon/openpi
ARG OPENPI_COMMIT=e95e1c6186ed16d55568043058fee95abaf4630c
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir --no-deps \
    "openpi-client @ git+${OPENPI_REPO}@${OPENPI_COMMIT}#subdirectory=packages/openpi-client"

# Assets — each subdirectory is a separate layer for better caching.
# Placed before code COPY so they stay cached across code changes.
# Ordered smallest to largest so cheaper layers come first.
COPY assets/robots/      /workspace/robolab/assets/robots/
COPY assets/materials/   /workspace/robolab/assets/materials/
COPY assets/fixtures/    /workspace/robolab/assets/fixtures/
COPY assets/backgrounds/ /workspace/robolab/assets/backgrounds/
COPY assets/scenes/      /workspace/robolab/assets/scenes/
COPY assets/objects/     /workspace/robolab/assets/objects/

# Accept NVIDIA/Isaac Sim EULAs
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y

# Copy project code (assets/ excluded via .dockerignore)
COPY . /workspace/robolab
WORKDIR /workspace/robolab

# Install Python dependencies
# GIT_LFS_SKIP_SMUDGE=1 avoids pulling large LFS objects during build
RUN GIT_LFS_SKIP_SMUDGE=1 /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir -e .
