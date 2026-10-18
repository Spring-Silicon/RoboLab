# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the eval runner's argparse wiring against AppLauncher collisions.

The per-policy runners build their parser as::

    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)

IsaacLab 2.3's ``add_app_launcher_args`` raises ``ValueError`` if the parser
already declares a field it owns (e.g. ``rendering_mode``). This never shows up
in the other tests because they drive ``AppLauncher`` directly rather than
through the runner's parser — so it's asserted explicitly here.
"""

import argparse
import os
import signal
import subprocess
import sys

from isaaclab.app import AppLauncher

from robolab.eval.runner import _claim_task, add_common_eval_args


def test_common_eval_args_do_not_collide_with_app_launcher():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    # Must not raise: AppLauncher owns `rendering_mode`, so add_common_eval_args
    # must not declare it (or any other AppLauncher-owned field).
    AppLauncher.add_app_launcher_args(parser)

    dests = {action.dest for action in parser._actions}
    # `renderer` and `rendering_type` are ours; `rendering_mode` is owned by
    # AppLauncher. run_evaluation() consumes args.renderer and args.rendering_type,
    # so both of ours must be present, and AppLauncher's field must coexist.
    assert "renderer" in dests
    assert "rendering_type" in dests
    assert "rendering_mode" in dests
    assert "enable_gt_state" in dests


def test_task_claim_excludes_until_released(tmp_path):
    # flock binds to the open file description, so a second open in this process
    # sees the claim as another --task-queue process would.
    first = _claim_task(str(tmp_path), "TaskA")
    assert first is not None
    assert _claim_task(str(tmp_path), "TaskA") is None
    other = _claim_task(str(tmp_path), "TaskB")
    assert other is not None
    first.close()  # a finished or crashed process releases its claims
    again = _claim_task(str(tmp_path), "TaskA")
    assert again is not None
    again.close()
    other.close()


def test_task_claim_released_when_holder_is_killed(tmp_path):
    lock = os.path.join(tmp_path, ".task_queue", "TaskA.lock")
    os.makedirs(os.path.dirname(lock))
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time\n"
            f"f = open({lock!r}, 'a')\n"
            "fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
            "print('held', flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert _claim_task(str(tmp_path), "TaskA") is None
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait()
    claim = _claim_task(str(tmp_path), "TaskA")
    assert claim is not None
    claim.close()
