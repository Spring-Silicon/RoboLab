# Camera gating: experimental, NOT accepted

Status: not merged into `perf/wave1` or `perf/sim-speed`. Kept here for reference only.
Do not turn it on for production runs.

## Code

- RoboLab branch `perf/camgate` (this branch): `robolab/eval/camera_gating.py`,
  `scripts/camgate_probe.py`, `scripts/camgate_episode_benchmark.py`, fake tests.
- Isaac Lab branch `perf/camgate` (commit `f45503e`): `TiledCamera` update toggling in
  `source/isaaclab/isaaclab/sensors/camera/tiled_camera.py` and `tiled_camera_cfg.py`,
  tests in `source/isaaclab/test/sensors/test_tiled_camera.py`.

## Tests

- Focused TiledCamera tests: 3 passed, 88 deselected (101.46 s).
- Full TiledCamera suite: 91 passed (190.22 s).
- RoboLab fake tests: 13 passed.

## A/B evidence (`camgate_ab_compare_chain3.json`)

- 48/48 non-image HDF5 datasets equal; joint position max difference 0.0.
- 57 policy query steps in each arm; 449 video frames in every file in both arms.
- No stale image in 57 checks per arm.

## Known regressions (reasons it is not accepted)

- Images change: same-step gated vs ungated mean pixel difference 7.2-7.6 (0-255).
  Two ungated runs differ by about 1.25, so gating changes images beyond RTX noise.
- Sensor-mode speed: 0.48x (gating is about 2.1x slower).
- Viewport mode: whole-episode speedup only 1.03x (step speedup about 1.35x).
- Only `rgb`, `rgba` and `distance_to_camera` support gating; other annotators raise `ValueError`.
