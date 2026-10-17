# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode timing includes work outside sections and counts useful batched steps."""

import pytest

from robolab.eval.episode import TimingStats


def test_wall_time_includes_reset_and_finalization(monkeypatch):
    clock = iter([10.0, 12.0, 15.0, 20.0])
    monkeypatch.setattr("robolab.eval.episode.time.perf_counter", lambda: next(clock))
    timer = TimingStats()
    timer.start("env_step")
    timer.stop("env_step")
    result = timer.to_dict(6, active_env_steps=15)
    assert result["wall_total_s"] == 10.0
    assert result["timed_sections_s"] == 3.0
    assert result["it_per_sec"] == pytest.approx(0.6)
    assert result["active_env_steps_per_sec"] == pytest.approx(1.5)
    assert result["control_steps"] == 6
    assert result["active_env_steps"] == 15


def test_short_runs_compute_rates_before_rounding(monkeypatch):
    clock = iter([1.0, 1.0001])
    monkeypatch.setattr("robolab.eval.episode.time.perf_counter", lambda: next(clock))
    result = TimingStats().to_dict(1, active_env_steps=2)
    assert result["wall_total_s"] == pytest.approx(0.0001)
    assert result["it_per_sec"] == pytest.approx(10_000)
    assert result["active_env_steps_per_sec"] == pytest.approx(20_000)
