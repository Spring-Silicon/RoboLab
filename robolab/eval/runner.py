# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evaluation driver: argparse helpers + per-task run loop.

Each per-policy script under ``policies/<policy>/run.py``:

1. Builds an :class:`argparse.ArgumentParser` with backend-specific flags.
2. Calls :func:`add_common_eval_args` to inject the shared flags.
3. Launches IsaacSim and registers envs (policy-specific camera presets etc.).
4. Defines a ``client_factory(args) -> InferenceClient`` closure.
5. Calls :func:`run_evaluation` with the policy name and the factory.

This module stays policy-agnostic — no backend names appear in it.

Import order constraint: :func:`add_common_eval_args` must be callable
*before* ``AppLauncher`` launches (so the parser knows the flags at parse
time). The heavy isaaclab/episode/summarize imports are therefore deferred
into :func:`run_evaluation`, which only runs *after* AppLauncher has set up
the simulation app.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Callable

from robolab.constants import DEFAULT_TASK_SUBFOLDERS

if TYPE_CHECKING:
    from robolab.eval.base_client import InferenceClient

    ClientFactory = Callable[[argparse.Namespace], InferenceClient]
else:
    ClientFactory = Callable[[argparse.Namespace], object]


def _unit_interval(s: str) -> float:
    v = float(s)
    if not (0.0 < v <= 1.0):
        raise argparse.ArgumentTypeError(f"--ci-pp-width must be in (0, 1], got {v}")
    return v


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared eval flags. Call this once per runner script."""
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1,
                        help="Number of environments to spawn.")
    parser.add_argument("--task", nargs="+", default=None,
                        help="List of tasks to evaluate on.")
    parser.add_argument("--tag", nargs="+", default=None,
                        help="List of tags of tasks to evaluate on.")
    parser.add_argument("--task-dirs", "--task_dirs", nargs="+", default=DEFAULT_TASK_SUBFOLDERS,
                        help="List of task directories to evaluate on.")
    parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                        help=("Number of sequential runs per task (default: 1). "
                              "Total episodes = num_runs * num_envs. "
                              "Prefer increasing --num-envs for more episodes. "
                              "Only increase --num-runs if you run out of GPU memory "
                              "with the desired num-envs."))
    parser.add_argument("--num-episodes-adaptive", "--num_episodes_adaptive",
                        type=int, default=None, metavar="MAX_N",
                        help=("Enable adaptive sampling per task. Overrides --num-runs. "
                              "Keeps adding batches of num_envs episodes until the 95%% "
                              "Beta credible interval on success rate is <= --ci-pp-width "
                              "wide, or until MAX_N episodes are reached. Recommended "
                              "MAX_N=200 (matches TRI LBM sim protocol, arXiv:2507.05331)."))
    parser.add_argument("--ci-pp-width", "--ci_pp_width", type=_unit_interval,
                        default=0.14, metavar="W",
                        help=("Target 95%% Beta credible interval width (as a fraction in "
                              "(0, 1]) for adaptive sampling. Default 0.14 = worst-case CI "
                              "width at n=200 (TRI LBM sim protocol, arXiv:2507.05331). "
                              "Only used when --num-episodes-adaptive is set."))
    parser.add_argument("--enable-subtask", "--enable_subtask", dest="enable_subtask",
                        action="store_true", default=True,
                        help="Enable subtask progress checking (default: on; kept for "
                             "backward compatibility).")
    parser.add_argument("--disable-subtask", "--disable_subtask", dest="enable_subtask",
                        action="store_false",
                        help="Disable subtask progress checking (episode results will "
                             "have no score/reason and an empty events log).")
    parser.add_argument("--enable-gt-state", "--enable_gt_state", action="store_true",
                        help="Export simulator ground-truth state to the policy client.")
    parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                        help=("Output folder name under <repo>/output. Default is "
                              "<timestamp>_<policy>. If you provide the output folder name "
                              "for a previous run, the script will skip the tasks and "
                              "episodes that have already been run."))
    parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default",
                        help=("Which instruction variant to use when a task defines multiple "
                              "(default, vague, specific, etc.)."))
    parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                        choices=["all", "viewport", "sensor", "none"],
                        help=("Which videos to save: 'all' (sensor + viewport), "
                              "'viewport' only, 'sensor' only, or 'none' (default: all)."))
    parser.add_argument("--render-policy-steps-only", "--render_policy_steps_only", action="store_true",
                        help=("Render cameras and compute image observations only on control steps whose "
                              "observation the client queries its policy with (action-chunking clients "
                              "with open_loop_horizon > 1). Physics, non-image observations, recorders and "
                              "success checks still run every step. Requires --video-mode none."))
    parser.add_argument("--render-warmup-steps", "--render_warmup_steps", type=int, default=0,
                        help=("With --render-policy-steps-only, also render (without camera read-back) this "
                              "many steps before each consumed step, so RTX temporal history matches rendering "
                              "every step more closely; >= open_loop_horizon - 1 renders every step and gives "
                              "the default pixels (default: 0)."))
    parser.add_argument("--task-queue", "--task_queue", action="store_true",
                        help=("Share the task list with other processes started with the same arguments and "
                              "--output-folder-name. Each process claims one task at a time with a file lock "
                              "and skips tasks that another process holds or finished, so K processes split "
                              "the tasks and each pays Kit startup once. Requires --output-folder-name and a "
                              "local file system (flock)."))
    parser.add_argument("--queue-unit", "--queue_unit", choices=("task", "run"), default="task",
                        help=("With --task-queue: claim whole tasks (default) or single runs. With 'run', "
                              "several processes split the runs of one task, so a long task with "
                              "--num-envs 1 runs its episodes in parallel. Each process starts with a fresh "
                              "environment, so a run's initial state does not depend on the runs before it."))
    parser.add_argument("--renderer", type=str, default="realtime",
                        choices=["realtime", "pathtracing"],
                        help=("RTX renderer mode (default: realtime). 'realtime' uses "
                              "the RaytracedLighting interactive renderer; 'pathtracing' "
                              "uses the offline path tracer (higher fidelity but much "
                              "slower — intended for beauty/demo renders, not large-N "
                              "eval). Sets carb '/rtx/rendermode'."))
    # NOTE: named --rendering-type (not --rendering-mode) on purpose. Isaac Lab's
    # AppLauncher.add_app_launcher_args() reserves the dest `rendering_mode` on some
    # versions (e.g. 2.2.0 in the OSMO eval image) and raises if the parser already
    # defines it. Using a distinct dest avoids that collision regardless of arg order.
    parser.add_argument("--rendering-type", "--rendering_type", type=str, default=None,
                        choices=["performance", "balanced", "quality"],
                        help=("Realtime renderer quality preset (maps to IsaacLab "
                              "RenderCfg.rendering_mode). Default: unset, which lets "
                              "IsaacLab fall back to 'balanced'. No effect under "
                              "--renderer pathtracing."))
    # Forwarded to Isaac Sim's SimulationApp config key of the same name through
    # AppLauncher (like --renderer). It sizes Kit's carb.tasking and TBB worker pools.
    parser.add_argument("--limit-cpu-threads", "--limit_cpu_threads", type=int, default=4,
                        help=("Kit worker-thread pool size (Isaac Sim 'limit_cpu_threads', default 32 there). "
                              "An eval process keeps only a few workers busy and idle default pools spin "
                              "several extra cores, so 4 keeps step time while letting more processes "
                              "share a machine. Only thread scheduling changes."))
    parser.add_argument("--policy-query-mode", "--policy_query_mode", type=str, default="auto",
                        choices=["auto", "serial", "concurrent", "batch"],
                        help=("How to send the policy queries of envs that need a new action chunk at the same "
                              "step. 'auto' (default): 'concurrent' only for clients whose server output does "
                              "not depend on request order (none of the shipped policy servers), else 'serial'. "
                              "'serial': the old loop, one env after the other. 'concurrent': one request per "
                              "env, all sent at the same time over separate connections; a server with a "
                              "shared RNG (pi0, GR00T, Cosmos) then gives each env other noise samples than "
                              "'serial'. 'batch': one request for all envs (GR00T); the server output can "
                              "differ from per-env requests. Clients that cannot use a mode fall back "
                              "(batch -> concurrent -> serial)."))
    parser.add_argument("--policy-query-workers", "--policy_query_workers", type=int, default=16,
                        help="Maximum concurrent policy connections in 'concurrent' mode (default: 16).")


def run_evaluation(
    args: argparse.Namespace,
    *,
    policy: str,
    client_factory: ClientFactory,
) -> None:
    """Drive the per-task evaluation loop.

    Must be called *after* ``AppLauncher`` has launched, since this is when
    the isaaclab-dependent modules below become safe to import.

    Args:
        args: Parsed argparse namespace. Must include the flags from
            :func:`add_common_eval_args` plus ``device`` and ``headless`` from
            ``AppLauncher.add_app_launcher_args``.
        policy: Backend label stamped into the output folder name and
            ``env_cfg.policy``. No behavioral branching.
        client_factory: Callable that builds the :class:`InferenceClient` given
            ``args``. Called once per task; the client is reused across runs
            within a task.
    """
    if args.render_policy_steps_only and args.video_mode != "none":
        raise ValueError("--render-policy-steps-only requires --video-mode none")
    if args.task_queue and args.output_folder_name is None:
        raise ValueError("--task-queue requires --output-folder-name (all processes must share one output folder)")
    if args.queue_unit == "run" and (not args.task_queue or args.num_episodes_adaptive is not None):
        raise ValueError("--queue-unit run requires --task-queue and a fixed --num-runs")
    if args.render_warmup_steps and not args.render_policy_steps_only:
        raise ValueError("--render-warmup-steps requires --render-policy-steps-only")
    import os

    import robolab.constants
    from robolab.constants import PACKAGE_DIR, get_timestamp, set_output_dir
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.logging.results import (
        check_all_episodes_complete,
        check_run_complete,
        init_experiment,
        load_episode_results,
        summarize_experiment_results,
    )
    from robolab.core.utils.adaptive_sampling import count_task_episodes, should_continue_sampling
    from robolab.core.utils.print_utils import print_experiment_summary
    from robolab.eval.episode import run_episode
    from robolab.eval.summarize import summarize_run

    if args.output_folder_name is not None:
        output_folder_name = args.output_folder_name
    else:
        output_folder_name = get_timestamp() + f"_{policy}"
        if args.instruction_type != "default":
            output_folder_name += f"_{args.instruction_type}"

    output_dir = os.path.join(PACKAGE_DIR, "output", output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.task:
        task_envs = get_envs(task=args.task)
        filter_str = f"tasks: {', '.join(args.task)}"
    elif getattr(args, "tag", None):
        task_envs = get_envs(tag=args.tag)
        filter_str = f"tags: {', '.join(args.tag)}"
    else:
        task_envs = get_envs()
        filter_str = "all"

    num_envs = args.num_envs
    num_runs = args.num_runs
    adaptive_max = args.num_episodes_adaptive
    is_adaptive = adaptive_max is not None
    total_episodes = adaptive_max if is_adaptive else num_runs * num_envs

    print_experiment_summary(
        task_envs=task_envs,
        filter_str=filter_str,
        num_envs=num_envs,
        num_episodes=total_episodes,
        policy=policy,
        instruction_type=args.instruction_type,
        output_dir=output_dir,
    )

    episode_results_file, episode_results = init_experiment(output_dir)

    save_videos = args.video_mode != "none"

    # Claims stay locked until this function returns (the process then exits): other
    # processes skip a claimed task, and a crash releases its claims for a later resume.
    task_claims = []

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        run_claims = args.task_queue and args.queue_unit == "run"
        if args.task_queue and not run_claims:
            claim = _claim_task(output_dir, task_env)
            if claim is None:
                print(f"\033[96m[RoboLab] Task `{task_env}` claimed by another process. Skipping.\033[0m")
                continue
            task_claims.append(claim)
        if args.task_queue:
            # Other processes append to the same results file; see their finished runs.
            episode_results = load_episode_results(output_dir)

        if check_all_episodes_complete(
            episode_results=episode_results, env_name=task_env, num_episodes=total_episodes
        ):
            print(f"\033[96m[RoboLab] Task `{task_env}` already done. Skipping.\033[0m")
            continue

        def run_done(run_idx: int) -> bool:
            return all(
                check_run_complete(episode_results=episode_results, env_name=task_env, episode=run_idx * num_envs + eid)
                for eid in range(num_envs)
            )

        def next_run(after: int) -> int | None:
            """The next unfinished run from ``after``; under --queue-unit run, also claim it."""
            nonlocal episode_results
            for run_idx in range(after, num_runs):
                if run_done(run_idx):
                    continue
                if run_claims:
                    claim = _claim_task(output_dir, f"{task_env}.run{run_idx}")
                    if claim is None:
                        continue
                    # A process that finished this run and exited has released its claim.
                    episode_results = load_episode_results(output_dir)
                    if run_done(run_idx):
                        claim.close()
                        continue
                    task_claims.append(claim)
                return run_idx
            return None

        if is_adaptive:
            run_idx = 0
        else:
            run_idx = next_run(0)
            if run_idx is None:
                print(f"\033[96m[RoboLab] Task `{task_env}`: every open run is claimed. Skipping.\033[0m")
                continue

        env, env_cfg = create_env(
            task_env,
            device=args.device,
            num_envs=num_envs,
            instruction_type=args.instruction_type,
            policy=policy,
            renderer=args.renderer,
            rendering_mode=args.rendering_type,
        )

        if robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING and getattr(env_cfg, "subtasks", None) is None:
            print(
                f"\033[93m[RoboLab] WARNING: Subtask tracking is enabled but task `{task_env}` "
                f"has no subtask specification — subtask tracking is skipped for this task, so "
                f"its episode results will have no score/reason and an empty events log.\033[0m"
            )

        client = client_factory(args)
        client.configure_queries(mode=args.policy_query_mode, max_workers=args.policy_query_workers)

        while True:
            if is_adaptive:
                k_so_far, n_so_far = count_task_episodes(episode_results, task_env)
                if not should_continue_sampling(
                    k=k_so_far, n=n_so_far, target_width=args.ci_pp_width, n_max=adaptive_max
                ):
                    print(
                        f"\033[96m[RoboLab] Task `{task_env}` adaptive stop at {n_so_far} "
                        f"episodes ({k_so_far}/{n_so_far} success).\033[0m"
                    )
                    break
                if run_done(run_idx):
                    print(f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m")
                    run_idx += 1
                    continue
            elif run_idx is None:
                break

            if args.instruction_type != "default":
                run_name = f"{task_env}_{args.instruction_type}_{run_idx}"
            else:
                run_name = f"{task_env}_{run_idx}"
            print(
                f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' "
                f"(run {run_idx}, {num_envs} envs)\033[0m"
            )

            env_results, msgs, timing = run_episode(
                env=env,
                env_cfg=env_cfg,
                episode=run_idx,
                client=client,
                save_videos=save_videos,
                video_mode=args.video_mode,
                headless=args.headless,
                enable_gt_state=args.enable_gt_state,
                render_policy_steps_only=args.render_policy_steps_only,
                render_warmup_steps=args.render_warmup_steps,
            )

            episode_results = summarize_run(
                env_results=env_results,
                msgs=msgs,
                env=env,
                env_cfg=env_cfg,
                num_envs=num_envs,
                run_idx=run_idx,
                run_name=run_name,
                task_env=task_env,
                scene_output_dir=scene_output_dir,
                policy=policy,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
                enable_subtask_progress=robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING,
                timing=timing,
                instruction_type=args.instruction_type,
            )

            env.reset_eval_state()
            run_idx = run_idx + 1 if is_adaptive else next_run(run_idx + 1)

        client.close()
        env.close()

    if args.task_queue:
        episode_results = load_episode_results(output_dir)
    summarize_experiment_results(episode_results, show_timing=True)


def _claim_task(output_dir: str, task_env: str):
    """Take a non-blocking exclusive lock for ``task_env``; return the open file, or None if held elsewhere."""
    import fcntl
    import os

    queue_dir = os.path.join(output_dir, ".task_queue")
    os.makedirs(queue_dir, exist_ok=True)
    claim = open(os.path.join(queue_dir, f"{task_env}.lock"), "a")
    try:
        fcntl.flock(claim.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        claim.close()
        return None
    return claim
