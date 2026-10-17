# Data Storage and Output

By default, proprio data will be recorded in the hdf5 file. To turn on image data recording, `robolab.constants.RECORD_IMAGE_DATA` needs to be set to True.

Recorded episodes can be played back — see [Replaying Recorded Episodes](replay.md).

## Single-Env vs Multi-Env Recording

RoboLab supports running multiple parallel environments per task. The recording system handles both cases:

**Single-env** (`--num_envs 1`, default):
- One episode per run, written to `run_{run_idx}.hdf5` as `demo_0`
- Videos: `{instruction}_{run_idx}.mp4`
- Subtask logs: `log_{run_idx}_env0.json`

**Multi-env** (`--num_envs N`):
- N parallel episodes per run, all written to `run_{run_idx}.hdf5` as `demo_0` through `demo_{N-1}`
- Each `demo_i` corresponds to `env_id=i` with independent trajectory data
- Videos: `{instruction}_{run_idx}_env{env_id}.mp4` (one per env)
- Subtask logs: `log_{run_idx}_env{env_id}.json` (one per env)
- Each env has its own subtask state machine, event tracker, and termination state
- Envs terminate independently — when an env succeeds, it freezes while others continue
- HDF5 recording uses auto-flush (every 500 steps) with concurrent per-env streaming — each env's buffer is flushed to its own demo without interfering with other envs
- Total episodes = `num_runs * num_envs`

> **Tip:** Prefer increasing `--num_envs` for more episodes. Only increase `--num-runs` (default 1) if you run out of GPU memory with the desired `num_envs`. For example, to run 24 episodes when only 12 fit in VRAM: `--num_envs 12 --num-runs 2`.

The `episode_results.jsonl` tracks each env independently with a unique `episode` ID (`run_idx * num_envs + env_id`) and an `env_id` field. Each line is a self-contained JSON object (JSONL format) for safe append-only writes.

## Output Directory Structure

```
output/
└── <output_folder>/
    ├── episode_results.jsonl
    ├── <ENV_NAME>/
        ├── run_0.hdf5                       # Run 0 data (demo_0..demo_{N-1} for N envs)
        ├── run_1.hdf5                       # Run 1 data (if num_runs > 1)
        ├── log_0_env0.json                  # Run 0, env 0 subtask log
        ├── log_0_env1.json                  # Run 0, env 1 subtask log
        ├── {instruction}_0_env0.mp4         # Run 0, env 0 observation video
        ├── {instruction}_0_env0_viewport.mp4
        ├── {instruction}_0_env1.mp4         # Run 0, env 1 observation video
        ├── {instruction}_0_env1_viewport.mp4
        └── env_cfg.json
```
For `ENV_NAME`, refer to [environment registration](environment_registration.md).

## HDF5 Data Structure

Each `run_{i}.hdf5` file contains the following hierarchical structure (`h5glance run_0.hdf5`)
```
└data (2 attributes)
  ├demo_0 (2 attributes)
  │ ├actions      [float32: 470 × 8]
  │ ├ee_pose
  │ │ ├position         [float32: 470 × 3]   # EE (base_link) position, relative to env origin
  │ │ ├orientation      [float32: 470 × 4]   # world-frame quaternion (wxyz)
  │ │ ├linear_velocity  [float32: 470 × 3]   # world frame
  │ │ └angular_velocity [float32: 470 × 3]   # world frame
  │ ├initial_state
  │ │ ├articulation
  │ │ │ └robot
  │ │ │   ├joint_position [float32: 1 × 13]
  │ │ │   ├joint_velocity [float32: 1 × 13]
  │ │ │   ├root_pose      [float32: 1 × 7]
  │ │ │   └root_velocity  [float32: 1 × 6]
  │ │ ├rigid_object
  │ │ │ ├banana
  │ │ │ │ ├root_pose      [float32: 1 × 7]
  │ │ │ │ └root_velocity  [float32: 1 × 6]
  │ │ │ ├bowl
  │ │ │ │ ├root_pose      [float32: 1 × 7]
  │ │ │ │ └root_velocity  [float32: 1 × 6]
  │ │ │ └rubiks_cube
  │ │ │   ├root_pose      [float32: 1 × 7]
  │ │ │   └root_velocity  [float32: 1 × 6]
  │ │ └cameras                              # one entry per camera sensor in the scene
  │ │   ├over_shoulder_left_camera
  │ │   │ ├position    [float32: 1 × 3]     # camera position, relative to env origin
  │ │   │ └orientation [float32: 1 × 4]     # world-frame quaternion (ROS, xyzw)
  │ │   └wrist_cam
  │ │     ├position    [float32: 1 × 3]
  │ │     └orientation [float32: 1 × 4]
  │ ├obs
  │ │ ├arm_joint_pos      [float32: 470]
  │ │ ├over_shoulder_left_camera  [uint8: 470 × 720 × 1280 × 3]
  │ │ ├gripper_pos        [float32: 470]
  │ │ └wrist_cam          [uint8: 470 × 720 × 1280 × 3]
  │ ├states
  │ │ ├articulation
  │ │ │ └robot
  │ │ │   ├joint_position [float32: 470 × 13]
  │ │ │   ├joint_velocity [float32: 470 × 13]
  │ │ │   ├root_pose      [float32: 470 × 7]
  │ │ │   └root_velocity  [float32: 470 × 6]
  │ │ └rigid_object
  │ │   ├banana
  │ │   │ ├root_pose      [float32: 470 × 7]
  │ │   │ └root_velocity  [float32: 470 × 6]
  │ │   ├bowl
  │ │   │ ├root_pose      [float32: 470 × 7]
  │ │   │ └root_velocity  [float32: 470 × 6]
  │ │   └rubiks_cube
  │ │     ├root_pose      [float32: 470 × 7]
  │ │     └root_velocity  [float32: 470 × 6]
  │ ├subtask
  │ │ ├completed  [uint8: 470]
  │ │ ├score      [float32: 470]
  │ │ └status     [uint16: 470]
  │ └bbox
  │   ├bbox_mm
  │   │ ├banana       [int16: 470 × 8 × 3]   # OBB corners in mm (relative to env origin)
  │   │ ├bowl         [int16: 470 × 8 × 3]
  │   │ └rubiks_cube  [int16: 470 × 8 × 3]
  │   └centroid
  │     ├banana       [float16: 470 × 3]      # centroid in meters (relative to env origin)
  │     ├bowl         [float16: 470 × 3]
  │     └rubiks_cube  [float16: 470 × 3]
  ├demo_1
    ....
```

## Data Structure Details

### Episodes
- **`demo_i`**: Episode data for `env_id=i`. In single-env mode, only `demo_0` exists. In multi-env mode, `demo_0` through `demo_{N-1}` exist in each `run_{run_idx}.hdf5`.

### Coordinate Frames
All recorded **positions** are in the **env-local frame** — relative to each env's
scene origin (`env.scene.env_origins`) — so trajectories are directly comparable across
parallel envs. This covers `ee_pose/position`, `*/root_pose` (initial_state and states),
`initial_state/.../cameras/position`, and `bbox` corners/centroid. All recorded
**orientations** and **velocities** are in the world frame; they are unaffected by the
static per-env translation.

### Data Components
- **`actions`**: Robot control commands (8-dimensional for joint positions)
- **`ee_pose`**: Per-step end-effector (`base_link`) pose and velocity
  - **`position`**: EE position, `(T, 3)`, env-local (relative to env origin)
  - **`orientation`**: EE orientation as a world-frame quaternion `(w, x, y, z)`, `(T, 4)`
  - **`linear_velocity`** / **`angular_velocity`**: EE velocities in world frame, `(T, 3)`
- **`initial_state`**: Starting configuration of robot, objects, and cameras
  - **`cameras/{name}`**: Per-camera extrinsics captured after reset/randomization — `position` (env-local, `(1, 3)`) and `orientation` (world-frame ROS quaternion `(x, y, z, w)`, `(1, 4)`). One entry per camera sensor in the scene.
- **`obs`**: Observations including joint positions, camera images, and gripper state
- **`states`**: Full state trajectory of robot and objects over time
- **`subtask`**: Task progress tracking metrics
- **`bbox`**: Per-step oriented bounding box data for all rigid objects
  - **`bbox_mm/{name}`**: OBB corners as `int16` in millimeters, shape `(T, 8, 3)`. Corner order: `[0-3]` bottom face (z-low), `[4-7]` top face (z-high). Positions are relative to env origin. Convert to meters: `corners_m = data[:] / 1000.0`
  - **`centroid/{name}`**: Object centroid as `float16` in meters, shape `(T, 3)`. Relative to env origin.

### Object Tracking
The system tracks three main objects:
- **`banana`**: Position and velocity in 3D space
- **`bowl`**: Container object state
- **`rubiks_cube`**: Manipulation target object

## File Descriptions

- **`run_{i}.hdf5`**: Per-run data file containing episode data for all envs in that run. Each `demo_j` within the file corresponds to `env_id=j`. Contains:
  - Robot observations (joint states, end-effector poses, etc.)
  - Actions taken by the policy
  - Task-specific metrics and success/failure indicators
  - Episode timestamps and metadata

- **`env_cfg.json`**: Configuration file containing the environment setup, including:
  - Task definition and parameters
  - Robot configuration
  - Observation and action space specifications
  - Scene and lighting settings

## Episode Results

#### `episode_results.jsonl`
The main results file, located at the top level of the output folder. Uses JSONL format (one JSON object per line) for safe append-only writes. Each line corresponds to one episode and contains task metadata, success/failure status, subtask scoring, trajectory metrics, and error events.

> **Note:** Legacy folders may contain `episode_results.json` (JSON array format). All analysis tools read both formats transparently.

```jsonl
{"env_name": "RubiksCubeAndBananaTask", "task_name": "RubiksCubeAndBananaTask", "run_name": "RubiksCubeAndBananaTask_1", "episode": 1, "policy": "pi05", "instruction": "Put the cube and the banana in the bowl", "instruction_type": "default", "attributes": ["conjunction", "simple"], "success": true, "score": 1.0, "reason": "Completed subtask 'pick_and_place' 1/1", "episode_step": 232, "duration": 15.467, "dt": 0.06666666666666667, "metrics": {"ee_sparc": -3.971, "joint_sparc_mean": -6.190, "ee_isj": 213.389, "joint_isj": 5408.055, "ee_path_length": 1.126, "joint_rmse_mean": 0.022, "ee_speed_max": 0.195, "ee_speed_mean": 0.068}, "events": {"TARGET_OBJECT_DROPPED": 4}}
```

**Fields:**
- `env_name`: Environment name (also the subfolder name on disk)
- `task_name`: Base task class name (multiple env_names can share the same task_name)
- `run_name`: Unique run identifier (`<ENV_NAME>_<RUN_IDX>`)
- `run`: Run index (which batch of parallel envs)
- `episode`: Global episode number (`run_idx * num_envs + env_id`)
- `env_id`: Which parallel environment within the run (0 to num_envs-1)
- `policy`: Policy backend used
- `instruction`: Language instruction for the task
- `instruction_type`: Instruction variant used (e.g., `default`, `vague`, `specific`)
- `attributes`: Task attribute tags (e.g., `simple`, `color`, `spatial`, `conjunction`)
- `success`: Whether the episode succeeded
- `score`: Subtask completion score (0.0 to 1.0), present when subtask tracking is enabled
- `reason`: Completion or failure reason, present when subtask tracking is enabled
- `episode_step`: Number of simulation steps in the episode
- `duration`: Episode duration in seconds
- `dt`: Simulation timestep (seconds per step)
- `metrics`: Trajectory quality metrics, computed automatically at the end of each episode from HDF5 data
  - `ee_sparc`: End-effector SPARC smoothness (more negative = smoother)
  - `joint_sparc_mean`: Mean SPARC across arm joints
  - `ee_isj`: End-effector Integrated Squared Jerk
  - `joint_isj`: Joint space Integrated Squared Jerk
  - `ee_path_length`: End-effector path length in meters
  - `joint_rmse_mean`: Mean joint tracking error (action vs actual)
  - `ee_speed_max`: Maximum end-effector speed (m/s)
  - `ee_speed_mean`: Mean end-effector speed (m/s)
- `timing`: One parallel run's timing, repeated on each environment's result row (do not sum across those rows)
  - `timing_schema_version`: `2`; older unversioned timings used the sum of selected sections as wall time
  - `policy_inference_s` / `policy_inference_avg_ms`: Host time in the client call and action assembly; includes observation extraction, chunk-cache work, and any policy/network wait
  - `env_step_s` / `env_step_avg_ms`: Host time inside `env.step()` (physics, rendering, observations, task checks, recording). Asynchronous GPU work may finish in a later section; these are not GPU kernel timings
  - `video_write_s` / `video_write_avg_ms`: Host time submitting frames to video writers; final encoder release is included in wall time, not this section
  - `timed_sections_s`: Sum of named host sections; excludes untimed work
  - `wall_total_s`: Actual elapsed run time from before resets through writer release and client reset, with CUDA synchronization at run boundaries. Excludes environment construction and `summarize_run()`; measure those separately for complete evaluation latency
  - `control_steps`: Number of completed batched `env.step()` calls
  - `active_env_steps`: Sum of active environment counts before each completed control step; excludes already-frozen environments
  - `it_per_sec`: Batched control steps per second (`control_steps / wall_total_s`)
  - `active_env_steps_per_sec`: Active environment steps per second (`active_env_steps / wall_total_s`)
  - Values retain clock precision; compare only runs with the same workload, versions, output settings, and hardware
- `events`: Error event counts, extracted from episode log files at the end of each episode
  - `WRONG_OBJECT_GRABBED`: Robot grabbed an unintended object
  - `GRIPPER_HIT_TABLE`: Gripper made contact with table
  - `GRIPPER_HIT_OBJECT`: Gripper collided with an object
  - `GRIPPER_FULLY_CLOSED`: Gripper closed fully (empty grasp)
  - `TARGET_OBJECT_DROPPED`: Target object was grabbed but dropped mid-transport
  - `MULTIPLE_OBJECTS_GRABBED`: Multiple objects grabbed simultaneously
  - `OBJECT_BUMPED`: Non-target object was nudged
  - `OBJECT_MOVED`: Non-target object was significantly displaced
  - `OBJECT_OUT_OF_SCENE`: Object fell off table or moved outside workspace
  - `wrong_objects_grabbed`: List of wrong object names that were grabbed

## Measuring simulator throughput

`scripts/benchmark_speed.py` measures a fixed-action **segment**, not policy
quality or complete evaluation latency. Run it in the Isaac Lab Python environment:

```bash
python scripts/benchmark_speed.py --headless --device cuda:0 \
  --task MustardInRightBinTask --seed 1 --num-envs 1 \
  --warmup-steps 20 --steps 100 --repeats 3 \
  --output /tmp/robolab-speed/baseline.json
```

The objective is median synchronized wall milliseconds per batched control step.
For multiple environments, also report aggregate environment steps/second; do not
confuse higher aggregate throughput with lower step latency. The default segment
advances 6.67 simulated seconds after 20 warm-up steps, without changing the
task's original episode horizon, 120 Hz physics, or 15 Hz control/render cadence.

Each repetition independently resets and warms up. An additional priming warm-up
is discarded. A termination, frozen environment, early artifact reset, nonfinite
state/observation, blank camera output, or missing arm motion invalidates the run.
Physics/render call counts and camera frame progression are recorded. The timed
loop includes lightweight validation reductions and preallocated trajectory copies.
It excludes policy/network work, video encoding, artifact hashing/writing, and
episode-end export; configured recorder streaming flushes remain inside the loop.

The JSON records resolved configuration, source hashes, runtime versions, effective
Kit controls, GPU identity/clocks, and directly referenced asset hashes. Transitive
USD dependencies are not fully hashed. Its sibling `<stem>_artifacts` directory
contains action bytes, trajectories, and start/end observation arrays. Preserve
both. Diagnostic `--phase-timing` and `--cprofile` runs attribute host costs but
are not interchangeable with uninstrumented throughput measurements.

```bash
python scripts/compare_speed.py /tmp/robolab-speed/baseline.json \
  /tmp/robolab-speed/candidate.json --output /tmp/robolab-speed/comparison.json
```

The comparator verifies artifact checksums, action identity, workload/configuration,
hardware/runtime compatibility, and saved state/image differences. Its default
tolerances are zero. GPU contacts and RTX rendering need not be bitwise repeatable;
measure baseline-versus-baseline variation and specify application acceptance
criteria before accepting numerical or pixel differences. It reports max/mean
absolute differences and changed-pixel fractions; passing sampled comparisons is
**not** proof of policy-success equivalence or of every intermediate camera frame.
Source changes and execution controls are reported for review. Isolate the GPU
from other workloads and compare complete runs, not partial output from failures.

`--device cpu` permits a diagnostic comparison of CPU PhysX with GPU RTX rendering.
It is a different physics backend, not automatically equivalent to GPU PhysX.
Likewise, camera ablations and altered rendering presets are different workloads.
The strict comparator rejects changed workload configurations even if they run faster.

For end-to-end evaluation, separately time `create_env()`, `run_episode()`, and
`summarize_run()` with the same actions, task horizon, outputs, and seeds. Verify
termination steps, scores, trajectory recordings, and encoded frame counts. Repeat
on a second task and at the intended environment batch size before generalizing.

Headless environments created through `create_env()` disable Kit's main-loop
wall-clock rate limiter after initialization. Physics stepping, camera cadence,
camera quality, and episode length are unchanged. GUI pacing is left unchanged.

### L40S measurements (Isaac Sim 5.0 / Isaac Lab 2.2.0)

Controlled measurements used RoboLab base commit
`9db0aaf09d9fe5d4f37b168320788258c7012463`, one L40S (driver 570.195.03),
an EPYC 9354 host with a reported 13.6-core container quota, and
`MustardInRightBinTask`. The segment used seed 1, 20 warm-up controls,
100 measured controls, all three production cameras, task checks and recorder
callbacks. Neither physics/solver settings nor camera quality/cadence were reduced.
These are measurements of this workload and host, not universal speed guarantees.

| GPU PhysX configuration | Environments | ms/batched control | Aggregate env-steps/s |
|---|---:|---:|---:|
| Original main-loop pacing | 1 | 166.55 | 6.00 |
| Headless pacing disabled | 1 | 122.60 | 8.16 |
| Original main-loop pacing | 4 | 169.17 | 23.65 |
| Headless pacing disabled | 4 | 165.50 | 24.17 |

Single-environment values are medians of three repetitions; four-environment
values use two. The one-environment improvement is 26.4% less segment wall time
(1.36x throughput). Four-environment ranges overlap: the observed 2.2% improvement
is not persuasive evidence of a gain. Batching improves aggregate throughput,
not individual control-step latency.

Two complete, 450-control scripted evaluations with sensor video, HDF5 export,
and summarization averaged 78.49 s before and 59.00 s after the pacing change
(24.8% less wall time). Environment construction and Kit startup are separate;
construction was approximately 11 s. Both versions recorded 450 samples and
timed out unsuccessfully, as expected for this non-solving action sequence.
This is not a learned-policy success-rate test.

All saved GPU trajectory samples were bit-identical across the pacing comparison.
RTX images were not: sampled mean absolute pixel differences were 0.044–0.122
on the 0–255 scale, versus 0.044–0.121 in a baseline-versus-baseline comparison.
This supports unchanged work, but is not a prespecified policy-equivalence test;
the zero-tolerance image comparison still fails and must not be relabeled passing.

The Nsight capture contained 786,240 CUDA kernels over 100 controls, with several
articulation kernels launching one 32-thread block. Captured CUDA durations summed
to 4.84 s. The capture warned about possible event loss and lacked Vulkan events:
it does not measure total rendering GPU time. This evidence motivates reducing
launch/synchronization overhead or improving batching before optimizing kernel
arithmetic. Native physics replacement remains a separate validation project.

#### CPU PhysX with RTX rendering (opt-in experiment)

The same L40S remains responsible for rendering with `--device cpu`. For the
single-environment fixed-action replay, CPU PhysX measured 131.15 ms/control
with 64 PyTorch threads, 83.97 ms with `OMP_NUM_THREADS=1`, and 63.22 ms after
vectorizing Isaac Lab's CPU camera-atlas unpacking. Each value uses three
repetitions. The camera change alone reduced CPU segment wall time by 24.7%.
Do not extrapolate the thread setting to different hosts or environment counts.
The device switch also moves state tensors and associated task/observation work
to the CPU. Its measured gain is not an isolated PhysX-solver speedup.

The native change belongs in Isaac Lab 2.2.0's
`source/isaaclab/isaaclab/sensors/camera/tiled_camera.py`: it replaces the CPU
per-pixel Warp loop with NumPy views and bulk copies into existing output buffers.
The CUDA path is unchanged. Applying the RoboLab patch alone does not install
this separate Isaac Lab patch, and RoboLab's default physics device remains CUDA.
The benchmark now includes sensor backend source hashes in its provenance.

Thirty native regression cases cover camera ordering, partial atlas rows,
channels and dtype conversion. A separate actual-render diagnostic compared
51 arrays (460,615,680 channel values) against the original CPU Warp kernel,
including a three-camera atlas with a padded tile: all values matched exactly.
All saved physics trajectories also matched the pre-retile CPU run exactly.
Independent rendered observations remain non-bitwise-repeatable; the strict
zero-tolerance end-to-end image comparison does not pass.

The two full CPU evaluations averaged 45.31 s before retiling and 35.15 s after,
versus 78.49 s for the original GPU configuration: 2.23x overall evaluation
throughput in this non-solving scripted workload. Across all eight evaluation
recordings (original GPU, unpaced GPU, CPU, optimized CPU), the 48 HDF5 dataset
schemas matched, each recording retained 450 samples, and each video decoded to
449 frames at 15 fps and 1280x360. The baseline also has 449 encoded frames;
this optimization does not remove a frame.

**CPU versus GPU PhysX is not an equivalent-backend claim.** With identical
segment actions, maximum robot joint-position differences were 2.33e-5 rad and
mustard-position differences were 3.97e-5 m; mustard angular velocity differed
by up to 0.0818 rad/s. This non-grasping sequence cannot establish contact or
policy-success equivalence. Keep the switch opt-in until paired manipulation
success/contact tests across tasks and seeds pass independently chosen criteria.
No timestep, solver iteration, collision fidelity, camera resolution, or episode
horizon reductions were used to obtain these measurements.

#### Final tuning, batching, and remaining bottlenecks

After the CPU camera change, four PyTorch threads outperformed one on this host.
With `OMP_NUM_THREADS=4` and the native patch, the final fixed-action results were:

| Task | Environments | Original GPU ms/control | Unpaced GPU ms/control | Optimized CPU ms/control | CPU aggregate env-steps/s |
|---|---:|---:|---:|---:|---:|
| MustardInRightBinTask | 1 | 166.55 | 122.60 | 58.47 | 17.10 |
| MustardInRightBinTask | 4 | 169.17 | 165.50 | 119.96 | 33.34 |
| BananaInBowlTask | 1 | 159.35 | 118.35 | 51.61 | 19.38 |

The first row uses three repetitions; the other rows use two. All CPU candidates
replayed the corresponding GPU action bytes. The GPU pacing comparison on the
second task also retained bit-identical saved trajectories. Changing CPU thread
count from one to four retained bit-identical saved trajectories at both tested
batch sizes. Cross-backend comparisons remain explicitly non-equivalent.

The final two complete Mustard evaluations took 28.96 and 28.52 s, averaging
28.74 s versus 78.49 s originally: 2.73x throughput, or 63.4% less wall time.
Both still produced 450 recorded samples and 449 decoded video frames, with the
same dataset schemas and video dimensions/rate. This remains a scripted workload
without policy/network latency, not a claim of 2.73x faster real-policy evaluation.
The segment improvement is 2.85x; do not substitute that for the full-run number.

In the optimized **one-thread CPU diagnostic profile**, inclusive host time per
control was approximately 24.5 ms in rendering, 17.0 ms in observation computation,
and 8.4 ms in physics stepping, against 66.4 ms total. CPU atlas unpacking itself
took about 0.57 ms/control. The observation manager computed twice per control.
These are instrumented host ranges, not isolated GPU durations; nested ranges must
not be summed. Even deleting the entire measured physics-step block would save
only about 13% of that profile's wall time. For this CPU candidate, investigate
observation materialization and rendering before attempting a new physics engine.
For the retained GPU backend, the small-grid launch overhead remains relevant.

Other measured trials did not justify production changes: eight Kit worker threads
gave 169.90 versus 170.09 ms/control with original pacing, and reducing PhysX GPU
buffer capacities gave 125.47 versus 123.12 ms/control in the unpaced comparison.
Those settings were not adopted. The later CPU-PhysX work below changed the
evaluation defaults for thread pools (`--limit-cpu-threads 4`) and the PhysX CPU
dispatcher (`worker_thread_count=0`) after measuring them.

To reproduce the CPU candidate after applying both source patches, retain the
GPU baseline's actions and run in the same Isaac Lab environment:

```bash
python -c 'import numpy as np; a=np.load("/tmp/robolab-speed/baseline_artifacts/actions.npz"); np.save("/tmp/robolab-speed/replay.npy", a["actions"])'
OMP_NUM_THREADS=4 python scripts/benchmark_speed.py --headless --device cpu \
  --task MustardInRightBinTask --num-envs 1 --seed 1 \
  --warmup-steps 20 --steps 100 --repeats 3 \
  --actions /tmp/robolab-speed/replay.npy \
  --output /tmp/robolab-speed/cpu-candidate.json
```

The comparator intentionally rejects this GPU-to-CPU comparison as equivalent
work: the physics device/configuration changed. Use its trajectory/image deltas
for qualification, not a permissive tolerance chosen after observing a speed gain.

#### Second pass: host work, thread pools and multi-process throughput

A second L40S host (driver 580.178.04, EPYC 9554 slice limited to 27.2 cores)
ran the original tree 25% slower than the first host, so every ratio below uses
the original measured on that same host: 207.64 ms/control for the segment and
100.59 s per complete Mustard evaluation (GPU PhysX, original pacing).

| Configuration (current tree) | Segment ms/control | Complete evaluation s | vs original |
|---|---:|---:|---|
| GPU PhysX | 82.75 | – | 2.51x segment; saved trajectories and non-image observations bit-identical to the original tree |
| CPU PhysX, `OMP_NUM_THREADS=4` | 25.57 | – | 8.12x segment |
| CPU PhysX, `OMP_NUM_THREADS=1` | 26.98 | 12.18 (12.23, 12.14) | 7.70x segment, 8.26x evaluation |

Each segment row uses three repetitions; the evaluation row two complete
450-control runs with sensor video and HDF5. Against the previous CPU-PhysX tree
on the same host, the CPU segment keeps every non-image saved array bit-identical,
and images differ by the RTX run-to-run spread (mean absolute 0.16–0.43 on 0–255).
The changes are: lazy contact-sensor refresh (`history_length=0`, RoboLab reads
only `force_matrix_w`), no duplicate pre-recorder observation pass, host-side
outdated-sensor tracking without a `nonzero()` per update, slice indexing for
contiguous actuator joint groups, an inline PhysX CPU dispatcher
(`worker_thread_count=0`), no per-step USD-to-Fabric residual sync, GPU atlas
unpacking with one pinned copy per camera on the CPU path, background video
encoding with frame preparation on the writer threads, batched recorder rows, and
Kit worker pools of 4 (`--limit-cpu-threads`). Under a contact-heavy replay the
lazy contact read gave identical subtask events and final states; contact forces
differed from the per-substep buffers by float rounding (at most 5.7e-6 N in
`force_matrix_w`), and on GPU the old per-substep buffers already differed from a
fresh PhysX read by the same order.

Per control at 1 env on CPU PhysX the remaining host time is mostly rendering:
about 7.4 ms of Kit/RTX frame preparation, about 6.4 ms blocked (not spinning)
waiting for the 7 ms GPU frame, then physics 3.4 ms, observations 3.1 ms and
recorders 2.5 ms. Turning off the render waits makes cameras return the previous
frame, so that overlap was not adopted.

Throughput on one L40S with several evaluation processes (full 450-control
episodes with sensor video and HDF5, CPU PhysX, `OMP_NUM_THREADS=1`):

| Processes | Episodes/hour | Busy cores | GPU util | GPU memory |
|---|---:|---:|---:|---:|
| original tree, 1 x 1 env | 35.8 | – | – | – |
| 8 x 1 env | 677 | 9.8 | 84% | 32.2 GiB |
| 1 x 10 envs | 584 | 2.3 | 40% | 9.2 GiB |
| 4 x 10 envs | 1342 | 5.8 | 92% | 37.0 GiB |

Processes sharing a GPU slow each other mainly through RTX: each process's 7 ms
frame stretches to about 17 ms at four 1-env processes as the GPU time-slices
between Vulkan contexts. Ten envs in one process render with 4.69 instead of
7.05 GPU-ms per env-step and need about 0.9 GiB per env instead of 4.0 GiB per
process, which is why 10-env processes give the highest throughput. Twelve 1-env
processes exceeded the 45 GiB the L40S provides and crashed the renderer. See
[Running Many Evaluations per Machine](environment_run.md#running-many-evaluations-per-machine).

For GPU PhysX at 1 env each physics substep runs about 640 small dependent CUDA
kernels (about 5.4 ms of GPU time), so GPU physics alone costs at least about
43 ms per control regardless of host speed; its cost stays flat up to 16 envs
(52 ms/control) while CPU PhysX grows about 1.6–2.2 ms per env.

## See Also

- [Analysis and Results Parsing](analysis.md) — Scripts for summarizing, comparing, and auditing experiment results
