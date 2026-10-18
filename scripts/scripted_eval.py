"""Run the real evaluation loop (robolab.eval.runner.run_evaluation) with a scripted client.

No policy server is used. The client sends the same fixed joint/gripper motion in every
episode, so physics, recordings and results depend only on the simulator. Use this script
to measure whole-eval wall time (Kit startup, env creation, episodes, close) and to compare
the output folders of different process layouts with scripts/compare_eval_outputs.py.

Usage (inside Isaac Lab's Python):
    python.sh scripts/scripted_eval.py --headless --device cpu --task A B C \\
        --video-mode sensor --output-folder-name /abs/out/NAME [--task-queue] \\
        --timing-output /abs/out/NAME_timing_<pid>.json

Results land in the --output-folder-name folder, in the same layout as policies/*/run.py.
The timing JSON has app launch, registration, per-task create/close/wall and total time.
It does not include Kit shutdown; time the whole process from outside for that.
"""

import time

T0 = time.perf_counter()
T0_UNIX = time.time()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import cv2  # noqa: F401,E402  (load before Kit, as policies/*/run.py do)
from isaaclab.app import AppLauncher  # noqa: E402

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--timing-output", required=True, help="Path of the timing JSON to write.")
parser.add_argument(
    "--settings-dump",
    default=None,
    help="Folder for full carb settings snapshots per task (TASK.create.json after create, TASK.close.json before close).",
)
add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app
T_APP = time.perf_counter()

import numpy as np  # noqa: E402

import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as runtime  # noqa: E402
from robolab.core.environments.env import RobolabEnv  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args.enable_subtask
auto_register_droid_envs(task_dirs=args.task_dirs, task=args.task)
T_REG = time.perf_counter()

task_records = []
_create_env = runtime.create_env
_close_env = RobolabEnv.close


def dump_settings(scene, stage):
    if args.settings_dump is None:
        return
    import carb.settings

    os.makedirs(args.settings_dump, exist_ok=True)
    tree = carb.settings.get_settings().get_settings_dictionary("/").get_dict()
    with open(os.path.join(args.settings_dump, f"{scene}.{stage}.json"), "w") as f:
        json.dump(tree, f, indent=1, sort_keys=True, default=repr)


def timed_create_env(scene, **kwargs):
    start = time.perf_counter()
    env, env_cfg = _create_env(scene, **kwargs)
    task_records.append({"task": scene, "start_s": start - T0, "create_s": time.perf_counter() - start})
    dump_settings(scene, "create")
    return env, env_cfg


def timed_close(self):
    dump_settings(task_records[-1]["task"], "close")
    start = time.perf_counter()
    _close_env(self)
    end = time.perf_counter()
    record = task_records[-1]
    record.update(close_s=end - start, end_s=end - T0, task_wall_s=end - T0 - record["start_s"])


runtime.create_env = timed_create_env
RobolabEnv.close = timed_close


class ScriptedClient:
    """Fixed motion: sine on joint 0, gripper toggles every 50 control steps."""

    def __init__(self):
        self.reset()

    def configure_queries(self, **_):
        pass

    def close(self):
        pass

    def begin_episode(self, episode_idx):
        self.reset()

    def reset(self, *, env_id=None):
        self.step = 0
        self.initial = None

    def controls_until_observation(self, env_id):
        return 1

    def infer_batch(self, obs, instruction, *, env_ids):
        if self.initial is None:
            arm = obs["proprio_obs"]["arm_joint_pos"].detach().cpu().numpy()
            self.initial = np.concatenate([arm, np.zeros((arm.shape[0], 1), dtype=arm.dtype)], axis=1)
        actions = self.initial.copy()
        actions[:, 0] += 0.04 * np.sin(self.step * 0.05)
        actions[:, 7] = float((self.step // 50) % 2)
        self.step += 1
        return {eid: {"action": actions[eid], "viz": None} for eid in env_ids}


run_evaluation(args, policy="scripted", client_factory=lambda _: ScriptedClient())
T_END = time.perf_counter()

payload = {
    "pid": os.getpid(),
    "start_unix": T0_UNIX,
    "app_launch_s": T_APP - T0,
    "register_s": T_REG - T_APP,
    "process_eval_s": T_END - T0,
    "tasks": task_records,
    "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))},
}
with open(args.timing_output, "w") as f:
    json.dump(payload, f, indent=1)
print("SCRIPTED_EVAL_DONE", json.dumps(payload), flush=True)
app.close()
