# Changelog

## [Unreleased]

### Added

- Fixed-action simulator segment benchmark with warm-up/repetition boundaries,
  source/configuration provenance, invalid-work rejection, and saved trajectory
  and camera evidence. An artifact comparator separates speed from changed work
  or numerical/visual divergence (`scripts/benchmark_speed.py`,
  `scripts/compare_speed.py`; usage in `docs/data.md`).
- Benchmark provenance includes upstream sensor implementation hashes and host
  thread controls. Comparisons report changed upstream source manifests. Documented
  L40S measurements distinguish the unchanged-GPU pacing fix from opt-in CPU
  PhysX and the separately patched Isaac Lab CPU camera path.
- Opt-in render-on-demand evaluation for action-chunking clients:
  `--render-policy-steps-only` (`run_episode(render_policy_steps_only=True)`) renders,
  reads back cameras and computes camera observation groups only on control steps
  whose observation the client queries its policy with
  (`InferenceClient.controls_until_observation`). Other steps return
  `UnrenderedObservationGroup` placeholders that raise if read, so no stale image
  reaches a policy; physics, non-image observations, recorders and success checks
  still run every step. Video groups (`image_obs` for sensor video, `viewport_cam`
  for viewport video) render every step, so videos keep all frames; the other
  cameras' render products are disabled on skipped steps
  (`TiledCamera.set_render_product_enabled`). Recorded policy observations
  (`--record-image-data`) hold zero rows for skipped camera groups and an
  `obs/camera_rendered/<group>` mask. The default (render every step) is unchanged.
  RTX temporal history makes skipped-step images differ from rendering every step;
  `--render-warmup-steps N` renders the `N` preceding steps (no read-back).
  Measurements in `docs/environment_run.md`.
- `--limit-cpu-threads` (default 4) for evaluation runners and `scripts/benchmark_speed.py`,
  forwarded to Isaac Sim's `limit_cpu_threads` to size Kit's carb.tasking and TBB worker pools
  (Isaac Sim's default is 32). One eval process keeps only a few workers busy; the default pools
  spin about four extra cores per process without shortening a step. Pass 32 to restore Isaac
  Sim's sizing. Only thread scheduling changes.

### Changed

- Contact sensors created by RoboLab default to `history_length=0`. RoboLab
  predicates, event tracking and recorders read only the latest `force_matrix_w`,
  so sensors now refresh lazily once per query instead of after every physics
  substep; latest contact values are unchanged. Pass a positive history to the
  `create_*contact_sensor_cfg` helpers for custom consumers of `*_history`.
- RoboLab recorder terms declare `reads_step_observations = False`, letting Isaac
  Lab skip the duplicate pre-recorder observation compute. Returned observations,
  recorded datasets and camera frame counts are unchanged.
- Sensor videos: half-scale frames use an exact integer 2x2 mean (bit-identical to
  the previous bilinear interpolation), computed for all active envs in one batch with
  one host transfer (`unpack_combined_frames`), and `VideoWriter` encodes on a
  background thread through an ordered, bounded queue; every frame is still encoded
  in order. Encoding errors surface from the next `write()` or from `release()`.
- On CPU, unpacked camera `rgb` observations (a strided view of the camera's `rgba`
  buffer) are packed with OpenCV before the observation manager's copy; values are
  unchanged.
- Subtask recording reads frozen-env state from host-side mirrors and transfers
  episode lengths/event masks once per step instead of once per environment.
- With CPU-device observations, sensor-video frames are prepared on each env's
  `VideoWriter` thread instead of the simulation thread: `run_episode` queues the
  env's camera images (`camera_frame_sources`) with the new
  `VideoWriter.write_deferred()`, and the writer computes the same exact half-scale
  combined frame (`combine_camera_frames`, one TorchScript call that does not hold
  the GIL). Encoder inputs are bit-identical and keep their order; errors preparing a
  frame surface like encoding errors. Policy clients must not modify observation
  tensors in place (no in-repo client does). GPU-device runs keep the batched path.
- `RobolabRecorderManager` snapshots each per-control recorder output once for all
  envs and splits it into per-env `EpisodeData` only when the episode buffers are
  read (flush, export, reset callbacks, `get_episode`) or the recorded env set
  changes, instead of one `EpisodeData.add` per env, leaf and control. Exported
  HDF5 datasets, attributes and sample counts are unchanged.

### Fixed

- Episode timing now measures reset-to-finalization wall time rather than the sum
  of selected host sections. CUDA is synchronized at episode boundaries; versioned
  timing records distinguish batched control steps from active environment steps.
  Environment creation and summary generation remain outside the episode timer.
- Headless `create_env()` disables Kit's wall-clock frame limiter after scene
  initialization. Physics timestep, solver settings, camera cadence/quality,
  task checks, recording, and episode horizons remain unchanged; GUI pacing is
  not changed.

### Changed

- RoboLab environments run PhysX with an inline CPU task dispatcher
  (`PhysxCfg.worker_thread_count = 0`). Only task scheduling changes: on the L40S
  segment benchmark the saved trajectories are bitwise identical while
  `physics_step` drops from 12.1 to 4.4 ms/control (CPU PhysX) and from 58.1 to
  51.2 ms/control (GPU PhysX). The previous `num_threads` entry matched no
  Isaac Lab field and had no effect.

## [0.3.1] - 2026-08-11

### Added

- Kinova Gen3 (Robotiq 2F-85) fixed-base support: robot config, joint-position
  environment registrations, and an example runner
  (`examples/run_kinova_jointpos.py`).
  ([#23](https://github.com/NVlabs/RoboLab/pull/23))
- Floor-standing robot support: a robot config can declare the
  `root_z_above_ground` label (the reach of its lowest colliders below the
  root), and the environment factory places the root on each task scene's
  authored ground plane plus that offset (`docs/robots.md`).
- Per-scene ground-height lock (`tests/test_scene_ground.py`): the task table
  is a dynamic rigid body resting on the ground plane, so the authored ground
  height sets the tabletop height and is now locked per scene.

### Changed

- Every robot config must now declare the `ee_recorder_bodies` label mapping
  HDF5 channel names to the articulation bodies recorded for end-effector
  pose (`{}` opts out). **Breaking for custom robot configs**: environment
  generation fails with a `ValueError` naming the config until the label is
  added — see `docs/robots.md#end-effector-pose-recording`.
- Remaining task-scene ground planes aligned to the canonical -0.697 height;
  legacy scenes keep their original -0.65 ground to preserve replay
  compatibility with existing recordings.

## [0.3.0] - 2026-08-07

### Added

- Explicit coordinate-frame contract: end-effector observations are now
  published in the robot-root frame, with a compatibility shim for older
  recordings (`robolab/core/logging/frame_compat.py`) and a reference doc
  (`docs/frames.md`).
- Ground-truth state export during stepping: `--enable-gt-state` makes every
  runner emit a raw per-env world-state snapshot each step under
  `obs["gt_state"]` (`robolab.eval.GroundTruthStateExporter`; schema in
  `docs/environment_run.md`).
- VoLo policy backend (`policies/volo/`): a proxy wrapper over existing
  clients that adds depth observations, camera calibration, and ground-truth
  state metadata to the inference payload.
- Robot-owned scene fixtures with per-robot fixture selection
  (`robolab/core/environments/scene_fixture.py`): a robot config declares the
  table fixture it is mounted on (or none, for robots with their own base),
  and the environment factory swaps the task scene's default fixture
  accordingly.
- Multi-gripper support in the task conditional system: robots can declare
  named gripper groups (e.g. `"gripper": ["left", "right"]` on a bimanual
  robot), and success/failure predicates can target any member of a group or
  require several grippers simultaneously (see `docs/task_conditionals.md`
  and `docs/event_tracking.md`).
- Pull request template and documentation of accepted PR types in
  `CONTRIBUTING.md`.
- News section in the README and an ecosystem page (`docs/ecosystem.md`).

### Fixed

- Explicitly-named tasks now resolve correctly from scoped content-pack
  folders.

## [0.2.1] - 2026-07-20

### Added

- Faithful replay of recorded episodes: new `robolab/core/replay/` module with
  recorded env-config overlay, initial-state restore, and per-step state
  validation; driver script `examples/run_recorded.py`; user guide
  `docs/replay.md`. ([#19](https://github.com/NVlabs/RoboLab/issues/19))
- Batched inference support in the eval client interface.
- The pytest install-verification suite now ships in `tests/` (documented
  fresh-install check: `uv run pytest tests/`).

### Changed

- Subtask progress tracking is enabled by default during evaluation.
- Updated the GR00T N1.7 client.
  ([#15](https://github.com/NVlabs/RoboLab/pull/15))
- Overrode isaacsim's stale `websockets==12.0` pin so `uv lock` resolves a
  modern websockets for the policy clients.

### Fixed

- `WorldState` extras handling on IsaacLab 2.3 (`XformPrimView`).

## [0.2.0] - 2026-07-07

### Added

- IsaacSim 5.1 / IsaacLab 2.3 support alongside 5.0 / 2.2, selected via the
  mutually-exclusive `isaac50` / `isaac51` extras.
- Relative differential-IK environment registrations
  (`examples/run_rel_ik_demo.py`).
- Realtime/pathtracing RTX renderer selection via `--rendering-type`.
- Poly Haven attribution files for the curated backgrounds.

### Changed

- Bundled backgrounds trimmed to a curated indoor set. More backgrounds can be obtained from Poly Haven.
- End-effector pose position is recorded in the env-local frame.
- Docs: asset contribution guidelines, conditionals
  restructure; generalized VRAM sizing guide.

### Fixed

- HDF5 episode success is derived from `terminated` rather than a named
  termination term.

## [0.1.1] - 2026-05-31

### Added

- Cosmos3 policy client; refreshed DreamZero, GR00T, and Pi0-family clients.
- Results dashboard (`robolab-dashboard` CLI).
- `/robolab-scenegen` and `/robolab-taskgen` Claude Code skills.
- Convex-hull placement predicates, adaptive sampling, and DROID IK control.
- `CONTRIBUTING.md` and a `uv`-based install flow.

### Changed

- Per-policy reorganization: one runner per backend under
  `policies/<policy>/run.py`.
- Docs revamp: per-policy READMEs, SPDX headers, `THIRD_PARTY_NOTICES.md`.
- Episode videos encoded as streaming H.264 (libx264) so they play in
  browsers.

### Fixed

- Placement order enforced in then-sequenced tasks.
- Frozen-env data leaks in the episode recorder.

## [0.1.0] - 2026-04-10

Initial public release: 100+ benchmark tasks,
DROID and Franka embodiments, server-client policy evaluation (Pi0 family,
GR00T, DreamZero), and HDF5 episode recording with analysis tools.

[0.3.1]: https://github.com/NVlabs/RoboLab/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/NVlabs/RoboLab/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/NVlabs/RoboLab/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NVlabs/RoboLab/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/NVlabs/RoboLab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/NVlabs/RoboLab/releases/tag/v0.1.0
