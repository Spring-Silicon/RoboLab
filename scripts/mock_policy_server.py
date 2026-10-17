# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock policy server and client for measuring policy-query batching.

The server runs as its own process (like a real policy server) and answers
over ``multiprocessing.connection`` sockets, one thread per connection:

- ``infer``: one observation. Sleeps a latency drawn from [min, max] ms, then
  returns an action chunk.
- ``infer_batch``: several observations. Sleeps ONE latency draw for the whole
  batch, then returns one chunk per observation.

Latencies and chunks are deterministic functions of the request contents
(images, state and prompt), so the same observations give the same actions in
every query mode, and any change in an observation changes the actions.
``--max-parallel 1`` makes the server compute one request at a time (like a
single-threaded openpi or GR00T server); the default computes all
connections in parallel (like several server replicas or an async server).

Run the server::

    python scripts/mock_policy_server.py --port 6123 --min-ms 20 --max-ms 80

Client: ``MockPolicyClient`` (an ``InferenceClient``) in this module; it uses
only NumPy and the standard library on the server side.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client, Listener

import numpy as np

AUTHKEY = b"robolab-mock-policy"
ARM_DOF = 7


def request_digest(obs: dict) -> int:
    """64-bit digest of all request arrays and the prompt; seeds latency and chunk."""
    h = hashlib.blake2b(digest_size=8)
    for key in sorted(obs):
        value = obs[key]
        h.update(key.encode())
        h.update(value.encode() if isinstance(value, str) else np.ascontiguousarray(value).tobytes())
    return int.from_bytes(h.digest(), "little")


def latency_s(digest: int, min_ms: float, max_ms: float) -> float:
    frac = (digest % 1_000_003) / 1_000_002
    return (min_ms + frac * (max_ms - min_ms)) / 1000.0


def action_chunk(state: np.ndarray, digest: int, horizon: int) -> np.ndarray:
    """Small joint-position offsets around the current arm state; gripper open."""
    arm = np.asarray(state[:ARM_DOF], dtype=np.float32)
    phase = (digest % 6283) / 1000.0
    steps = np.arange(horizon, dtype=np.float32)[:, None]
    joints = np.arange(ARM_DOF, dtype=np.float32)[None, :]
    offsets = 0.03 * np.sin(phase + 0.3 * steps + 0.7 * joints)
    chunk = np.zeros((horizon, ARM_DOF + 1), dtype=np.float32)
    chunk[:, :ARM_DOF] = arm[None, :] + offsets
    return chunk


class MockPolicyServer:
    def __init__(self, port: int, min_ms: float, max_ms: float, horizon: int, max_parallel: int):
        self.listener = Listener(("127.0.0.1", port), authkey=AUTHKEY)
        self.min_ms, self.max_ms, self.horizon = min_ms, max_ms, horizon
        self.compute = threading.Semaphore(max_parallel) if max_parallel > 0 else None

    def _work(self, digest: int) -> None:
        if self.compute is None:
            time.sleep(latency_s(digest, self.min_ms, self.max_ms))
            return
        with self.compute:
            time.sleep(latency_s(digest, self.min_ms, self.max_ms))

    def _serve(self, conn) -> None:
        with conn:
            while True:
                try:
                    message = conn.recv()
                except EOFError:
                    return
                op = message["op"]
                if op == "infer":
                    obs = message["obs"]
                    digest = request_digest(obs)
                    self._work(digest)
                    conn.send({"actions": action_chunk(obs["state"], digest, self.horizon)})
                elif op == "infer_batch":
                    batch = message["obs"]
                    digests = [request_digest(obs) for obs in batch]
                    self._work(digests[0] ^ len(batch))
                    conn.send([{"actions": action_chunk(obs["state"], d, self.horizon)} for obs, d in zip(batch, digests)])
                elif op == "ping":
                    conn.send("pong")
                else:
                    conn.send({"error": f"unknown op {op!r}"})

    def serve_forever(self) -> None:
        print(f"MOCK_POLICY_READY {self.listener.address[1]}", flush=True)
        while True:
            conn = self.listener.accept()
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()


def start_server_process(port: int, min_ms: float, max_ms: float, horizon: int, max_parallel: int = 0):
    """Start the server in a child process (same interpreter); return it once it accepts connections."""
    proc = subprocess.Popen(
        [sys.executable, __file__, "--port", str(port), "--min-ms", str(min_ms), "--max-ms", str(max_ms),
         "--horizon", str(horizon), "--max-parallel", str(max_parallel)],
        stdout=subprocess.PIPE, text=True,
    )
    line = proc.stdout.readline()
    if not line.startswith("MOCK_POLICY_READY"):
        proc.kill()
        raise RuntimeError(f"mock policy server failed to start: {line!r}")
    return proc


try:
    from robolab.eval.base_client import InferenceClient
except ImportError:  # server-only use without robolab on the path
    InferenceClient = None

if InferenceClient is not None:

    class MockPolicyClient(InferenceClient):
        """Chunking client for the mock server, with pi0-like observation handling.

        Extracts both production camera images and the proprio state, sends the
        images at half resolution with the state, and replays the returned chunk.
        """

        supports_concurrent_queries = True
        # The chunk is a function of the request only, so "auto" sends queries concurrently.
        order_independent_server = True
        supports_batched_queries = True

        def __init__(self, port: int, open_loop_horizon: int = 10):
            super().__init__()
            self.open_loop_horizon = open_loop_horizon
            self._address = ("127.0.0.1", port)
            self.conn = self._open_connection()

        def _open_connection(self):
            return Client(self._address, authkey=AUTHKEY)

        def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
            images = raw_obs["image_obs"]
            proprio = raw_obs["proprio_obs"]
            return {
                "external_image": self._to_numpy(images["over_shoulder_left_camera"], env_id),
                "wrist_image": self._to_numpy(images["wrist_cam"], env_id),
                "state": np.concatenate([
                    self._to_numpy(proprio["arm_joint_pos"], env_id).astype(np.float32).reshape(-1),
                    self._to_numpy(proprio["gripper_pos"], env_id).astype(np.float32).reshape(-1),
                ]),
            }

        def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
            return {
                "external_image": np.ascontiguousarray(extracted_obs["external_image"][::2, ::2]),
                "wrist_image": np.ascontiguousarray(extracted_obs["wrist_image"][::2, ::2]),
                "state": extracted_obs["state"],
                "prompt": instruction,
            }

        def _query_server(self, request: dict) -> dict:
            return self._query_server_on(self.conn, request)

        def _query_server_on(self, connection, request: dict) -> dict:
            connection.send({"op": "infer", "obs": request})
            return connection.recv()

        def _query_server_batch(self, requests: list[dict]) -> list[dict]:
            self.conn.send({"op": "infer_batch", "obs": requests})
            return self.conn.recv()

        def _unpack_response(self, response: dict) -> np.ndarray:
            return np.asarray(response["actions"])

        def close(self) -> None:
            super().close()
            self.conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=6123)
    parser.add_argument("--min-ms", type=float, default=20.0)
    parser.add_argument("--max-ms", type=float, default=80.0)
    parser.add_argument("--horizon", type=int, default=10, help="Action chunk length.")
    parser.add_argument("--max-parallel", type=int, default=0,
                        help="Requests computed at the same time (0: no limit; 1: one at a time).")
    args = parser.parse_args()
    MockPolicyServer(args.port, args.min_ms, args.max_ms, args.horizon, args.max_parallel).serve_forever()


if __name__ == "__main__":
    main()
