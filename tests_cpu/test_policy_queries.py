# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the multi-env policy query modes of ``InferenceClient.infer_batch``.

No Isaac Sim: run with ``python -m pytest tests_cpu`` (``tests/conftest.py`` boots Isaac).
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from robolab.eval.base_client import InferenceClient  # noqa: E402

NUM_ENVS = 6
HORIZON = 4


class Unrendered(dict):
    unrendered = True


def make_obs(step: int, *, unrendered: bool = False) -> dict:
    state = np.arange(NUM_ENVS * 3, dtype=np.float32).reshape(NUM_ENVS, 3) + step
    image = np.full((NUM_ENVS, 2, 2), step, dtype=np.uint8)
    return {"state": {"q": state}, "image_obs": Unrendered() if unrendered else {"cam": image}}


class FakeClient(InferenceClient):
    """Deterministic in-process 'server': the chunk depends on the request only."""

    open_loop_horizon = HORIZON
    supports_concurrent_queries = True
    supports_batched_queries = True
    order_independent_server = True

    def __init__(self, delay_s: float = 0.0):
        super().__init__()
        self.delay_s = delay_s
        self.extracted: list[int] = []
        self.opened: list[dict] = []
        self.calls = {"serial": 0, "on": 0, "batch": 0}
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def _extract_observation(self, raw_obs, *, env_id=0):
        self.extracted.append(env_id)
        return {"q": raw_obs["state"]["q"][env_id].copy(), "img": raw_obs["image_obs"]["cam"][env_id].copy()}

    def _pack_request(self, extracted_obs, instruction):
        return {"q": extracted_obs["q"], "prompt": instruction}

    def _compute(self, request):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay_s)
        with self._lock:
            self.active -= 1
        base = float(request["q"].sum()) + len(request["prompt"])
        return {"actions": base + np.arange(HORIZON * 2, dtype=np.float64).reshape(HORIZON, 2)}

    def _query_server(self, request):
        self.calls["serial"] += 1
        return self._compute(request)

    def _open_connection(self):
        conn = {"closed": False, "thread": threading.get_ident()}
        self.opened.append(conn)
        return conn

    def _query_server_on(self, connection, request):
        # Each connection is used only on the thread that opened it (ZeroMQ sockets need this).
        assert connection["thread"] == threading.get_ident() != threading.main_thread().ident
        assert not connection["closed"]
        self.calls["on"] += 1
        return self._compute(request)

    def _close_connection(self, connection):
        assert connection["thread"] == threading.get_ident()
        connection["closed"] = True

    def _query_server_batch(self, requests):
        self.calls["batch"] += 1
        return [self._compute(request) for request in requests]

    def _unpack_response(self, response):
        return response["actions"]

    def _build_visualization(self, extracted_obs):
        return extracted_obs["img"]


def run(client: InferenceClient, steps: int = 13) -> list:
    """Episode-like loop; env 2 resets at step 5 so refreshes are staggered."""
    trace = []
    for step in range(steps):
        if step == 5:
            client.reset(env_id=2)
        env_ids = list(range(NUM_ENVS)) if step < 9 else [0, 2, 3]  # some envs frozen at the end
        rets = client.infer_batch(make_obs(step), "pick", env_ids=env_ids)
        trace.append([
            (env_id, rets[env_id]["action"].tolist(), rets[env_id]["viz"].tolist(),
             client.controls_until_observation(env_id))
            for env_id in sorted(rets)
        ])
    return trace


@pytest.mark.parametrize("mode", ["concurrent", "batch", "auto"])
def test_modes_match_serial_loop(mode):
    serial = FakeClient()
    serial.configure_queries(mode="serial")
    fast = FakeClient()
    fast.configure_queries(mode=mode, max_workers=4)

    assert run(fast) == run(serial)
    assert fast.extracted.count(2) == serial.extracted.count(2)
    assert sorted(fast.extracted) == sorted(serial.extracted)
    assert serial.calls["on"] == serial.calls["batch"] == 0
    assert fast.calls["on"] + fast.calls["batch"] > 0
    assert len(fast.opened) <= 4


def test_concurrent_queries_overlap_and_close():
    client = FakeClient(delay_s=0.05)
    client.configure_queries(mode="concurrent", max_workers=3)
    start = time.perf_counter()
    client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))
    elapsed = time.perf_counter() - start
    # 6 queries over 3 connections: 2 rounds of 50 ms, not 6.
    assert client.max_active == 3
    assert elapsed < 0.25
    assert len(client.opened) == 3
    client.close()
    assert all(conn["closed"] for conn in client.opened)
    assert client._query_threads == [] and client._query_connections == []


@pytest.mark.parametrize("mode", ["concurrent", "batch"])
def test_unordered_env_ids_match_serial_loop(mode):
    serial = FakeClient()
    serial.configure_queries(mode="serial")
    fast = FakeClient()
    fast.configure_queries(mode=mode, max_workers=2)
    env_ids = [4, 0, 5, 2]
    for step in range(2 * HORIZON + 1):
        obs = make_obs(step)
        expected = serial.infer_batch(obs, "pick", env_ids=env_ids)
        got = fast.infer_batch(obs, "pick", env_ids=env_ids)
        assert list(got) == env_ids
        for env_id in env_ids:
            np.testing.assert_array_equal(got[env_id]["action"], expected[env_id]["action"])
            assert fast.controls_until_observation(env_id) == serial.controls_until_observation(env_id)
    fast.close()


def test_failed_concurrent_query_waits_for_all_workers_and_reconnects():
    class Failing(FakeClient):
        def _query_server_on(self, connection, request):
            if self.calls["on"] == 0 and connection is self.opened[0]:
                self.calls["on"] += 1
                raise ConnectionError("lost")
            return super()._query_server_on(connection, request)

    client = Failing(delay_s=0.05)
    client.configure_queries(mode="concurrent", max_workers=3)
    with pytest.raises(ConnectionError):
        client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))
    # The error is raised only after the other workers finished, on closed connections.
    assert client.active == 0
    assert all(conn["closed"] for conn in client.opened)
    # No chunk was stored for the failed step: the retry queries every env on new connections.
    rets = client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))
    assert sorted(rets) == list(range(NUM_ENVS))
    assert len(client.opened) == 6
    client.close()


def test_auto_is_serial_unless_server_is_order_independent():
    class OrderDependent(FakeClient):
        order_independent_server = False

    client = OrderDependent()
    client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))
    assert client.calls == {"serial": NUM_ENVS, "on": 0, "batch": 0}


def test_fallbacks():
    class NoBatch(FakeClient):
        supports_batched_queries = False

    client = NoBatch()
    client.configure_queries(mode="batch")
    client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))
    assert client.calls == {"serial": 0, "on": NUM_ENVS, "batch": 0}

    class Plain(FakeClient):
        supports_concurrent_queries = False
        supports_batched_queries = False

    client = Plain()
    client.configure_queries(mode="batch")
    client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))
    assert client.calls == {"serial": NUM_ENVS, "on": 0, "batch": 0}

    class CustomInfer(FakeClient):
        def infer(self, obs, instruction, *, env_id=0):
            return super().infer(obs, instruction, env_id=env_id)

    client = CustomInfer()
    client.configure_queries(mode="concurrent")
    client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))
    assert client.calls == {"serial": NUM_ENVS, "on": 0, "batch": 0}

    client = FakeClient()
    client.configure_queries(mode="concurrent", max_workers=1)
    client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))
    assert client.calls == {"serial": NUM_ENVS, "on": 0, "batch": 0}


def test_unrendered_cached_envs_are_not_extracted():
    serial, fast = FakeClient(), FakeClient()
    serial.configure_queries(mode="serial")
    fast.configure_queries(mode="concurrent")
    results = []
    for client in (serial, fast):
        client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))
        client.reset(env_id=1)
        client.reset(env_id=4)
        client.infer_batch(make_obs(1), "pick", env_ids=list(range(NUM_ENVS)))  # refresh 1 and 4 only
        client.extracted.clear()
        # Unrendered step: every env has a cached chunk, so no env reads the cameras.
        rets = client.infer_batch(make_obs(2, unrendered=True), "pick", env_ids=list(range(NUM_ENVS)))
        assert client.extracted == []
        assert all(ret["viz"] is None for ret in rets.values())
        results.append([rets[e]["action"].tolist() for e in sorted(rets)])
    assert results[0] == results[1]
    assert fast.calls["on"] == 2 + NUM_ENVS


def test_batch_response_count_is_checked():
    class Short(FakeClient):
        def _query_server_batch(self, requests):
            return super()._query_server_batch(requests)[:-1]

    client = Short()
    client.configure_queries(mode="batch")
    with pytest.raises(RuntimeError, match="responses"):
        client.infer_batch(make_obs(0), "pick", env_ids=list(range(NUM_ENVS)))


def test_invalid_configuration():
    client = FakeClient()
    with pytest.raises(ValueError):
        client.configure_queries(mode="parallel")
    with pytest.raises(ValueError):
        client.configure_queries(max_workers=0)


def test_mock_server_actions_match_across_modes():
    mock = pytest.importorskip("mock_policy_server")
    port = 6400 + (int(time.time() * 1000) % 500)
    proc = mock.start_server_process(port, 1.0, 3.0, horizon=HORIZON)
    try:
        traces = {}
        for mode in ("serial", "concurrent", "batch"):
            client = mock.MockPolicyClient(port, open_loop_horizon=HORIZON)
            client.configure_queries(mode=mode)
            trace = []
            for step in range(9):
                obs = {
                    "image_obs": {
                        "over_shoulder_left_camera": np.full((NUM_ENVS, 8, 8, 3), step, dtype=np.uint8),
                        "wrist_cam": np.full((NUM_ENVS, 8, 8, 3), 2 * step, dtype=np.uint8),
                    },
                    "proprio_obs": {
                        "arm_joint_pos": np.arange(NUM_ENVS * 7, dtype=np.float32).reshape(NUM_ENVS, 7) * step,
                        "gripper_pos": np.zeros((NUM_ENVS, 1), dtype=np.float32),
                    },
                }
                rets = client.infer_batch(obs, "pick", env_ids=list(range(NUM_ENVS)))
                trace.append([rets[e]["action"].tolist() for e in sorted(rets)])
            client.close()
            traces[mode] = trace
        assert traces["concurrent"] == traces["serial"]
        assert traces["batch"] == traces["serial"]
    finally:
        proc.kill()
        proc.wait()
