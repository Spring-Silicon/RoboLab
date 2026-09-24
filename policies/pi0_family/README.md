# Pi0 Family (OpenPI)

The Pi0 family — `pi0`, `pi0_fast`, `pi05`, `pi05_compiled_regular`, `pi05_compiled_optimized`, `paligemma`, and `paligemma_fast` — is served by a single client, `Pi0DroidJointposClient`, over a WebSocket-based OpenPI policy server. The variant is selected at runtime via `--policy`; each variant supplies its own per-variant defaults inside the client.

See the [policies README](../README.md) for the shared client architecture and common CLI options.
For pi0-family variants, pass `--policy {pi0,pi0_fast,pi05,pi05_compiled_regular,pi05_compiled_optimized,paligemma,paligemma_fast}`.

## Install the server

1. Clone [`git@github.com:xuningy/openpi.git`](https://github.com/xuningy/openpi) and follow install instructions there. **Do not** install OpenPI in the same virtual environment as RoboLab — it runs separately.

2. Install the OpenPI **client** in the RoboLab environment:
   ```bash
   cd robolab
   uv pip install -e ../openpi/packages/openpi-client
   ```

## Start the policy server

Open a separate terminal and launch the server. We set `XLA_PYTHON_CLIENT_MEM_FRACTION` to 50% to avoid JAX consuming all GPU memory.

**Pi05:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

**Pi0-fast:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi0_fast_droid_jointpos
```

**Pi0:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi0_droid_jointpos
```

**PaliGemma Binning:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=paligemma_binning_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/paligemma_binning_droid_jointpos
```

## Compiled Pi05 servers

The Spring Silicon OpenPI fork provides `openpi-serve-compiled` for the two
Intel B580 deployment artifacts. Both expose the same OpenPI WebSocket contract
as Pi05 and return 15 absolute joint-position actions per request.

```bash
# Checkpoint-A OpenVINO W8A8 runtime.
openpi-serve-compiled \
    --variant pi05_compiled_regular \
    --artifact-dir /opt/pi05-compiled-regular-openvino-a \
    --host 0.0.0.0 --port 8000

# Checkpoint-A fused SYCL runtime.
openpi-serve-compiled \
    --variant pi05_compiled_optimized \
    --artifact-dir /opt/pi05-ssog-a-mc2-mux \
    --host 0.0.0.0 --port 8000
```

Only one server may own the B580 at a time. RoboLab validates the server's
`policy_id` and action horizon before evaluation, so a mislabeled run fails
before the simulator starts sending observations.

## Run evaluation

In the RoboLab terminal

```bash
cd robolab
uv run python policies/pi0_family/run.py --policy pi05 --headless
```

The default connection is `localhost:8000`. To change:
```bash
uv run python policies/pi0_family/run.py --policy pi05 --remote-host <HOST> --remote-port <PORT>
```

A full WebSocket URI (e.g. a hosted endpoint) can be passed with `--remote-uri`, which overrides `--remote-host` / `--remote-port`.

## Variation scripts

The pi0_family folder also ships controlled-variation runners that wrap the same client to sweep a single axis per registered env:

- `run_lighting.py` — lighting variations
- `run_camera_pose_variation.py` — camera pose variations
- `run_background_variation.py` — background variations
- `run_table_variation.py` — table variations

Each takes the same connection and common eval flags as `run.py`.
