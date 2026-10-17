# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is the source of truth. A verbatim copy lives at
# droid_plus/eval/base_client.py — keep both in sync when editing.

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

import numpy as np

# How ``infer_batch`` sends the policy queries of the envs that need a new
# action chunk at the same control step (see ``InferenceClient.configure_queries``).
QUERY_MODES = ("auto", "serial", "concurrent", "batch")


class InferenceClient(ABC):
    """Root client for policy inference.

    Subclass override surface, in order of increasing commitment:

    1. Implement the four hooks (``_extract_observation``, ``_pack_request``,
       ``_query_server``, ``_unpack_response``). Chunking, env-id bookkeeping,
       visualization, and reset are handled by the base.
    2. Additionally override ``_postprocess_chunk`` or ``_build_visualization``
       for action-space / logging quirks (gripper binarization, 7->8 padding).
    3. Override ``infer`` entirely if your flow isn't query-then-step-chunk
       (e.g. server-side session state, pre-step caching).

    Two hooks are meant to be split per concern:

      ``_extract_observation``  <- repo-specific (real-robot flat numpy dict vs
                                   sim nested torch batched dict)
      ``_pack_request``         <- backend-specific (wire keys, image sizes)

    Keeping these separate lets the same backend client be paired with
    different observation sources without duplicating the wire format.
    """

    # Subclasses override to match their server's chunk length.
    # horizon=1 is correct for single-action servers.
    open_loop_horizon: int = 1
    # Clients whose transport can open independent connections set this and
    # implement ``_open_connection`` / ``_query_server_on``.
    supports_concurrent_queries: bool = False
    # Set only when the server returns the same chunk for the same request,
    # independent of the order in which it gets requests (no shared RNG or
    # per-connection state). Then ``"auto"`` sends queries concurrently.
    order_independent_server: bool = False
    # Clients whose server accepts several observations in one request set
    # this and implement ``_query_server_batch``.
    supports_batched_queries: bool = False

    # Defaults for ``configure_queries``.
    query_mode: str = "auto"
    max_concurrent_queries: int = 16

    def __init__(self) -> None:
        # Per-env chunking state. Subclasses may ignore and manage state however
        # they want.
        self._chunks: dict[int, np.ndarray] = {}
        self._counters: dict[int, int] = {}
        # Set by begin_episode(); see below.
        self._eval_episode_idx: int = 0
        # Concurrent-query connections, opened on first use. Connection i is
        # opened, used and closed only on the thread of executor i.
        self._query_connections: list[Any] = []
        self._query_threads: list[ThreadPoolExecutor] = []

    def configure_queries(self, *, mode: str | None = None, max_workers: int | None = None) -> None:
        """Select how ``infer_batch`` sends the queries of several envs at one control step.

        - ``"auto"`` (default): ``"concurrent"`` for clients with
          ``order_independent_server``, else ``"serial"``. Actions are then the
          same as with the serial loop.
        - ``"serial"``: the historical loop over :meth:`infer`, one env at a time.
        - ``"concurrent"``: one request per env, sent at the same time over
          separate connections (at most ``max_workers``). Clients without
          ``supports_concurrent_queries`` use ``"serial"``. A server with a
          shared RNG (pi0, GR00T, Cosmos) then gives other noise samples to
          each env than the serial loop.
        - ``"batch"``: one request for all envs, for clients with
          ``supports_batched_queries``; other clients use ``"concurrent"``.
          The server computes all envs in one forward pass, so its outputs can
          differ from per-env requests (numerics, noise sampling).

        Each env gets the same observation and request in every mode, and the
        same action chunk if the server returns the same chunk for the same request.
        """
        if mode is not None:
            if mode not in QUERY_MODES:
                raise ValueError(f"query mode must be one of {QUERY_MODES}, got {mode!r}")
            self.query_mode = mode
        if max_workers is not None:
            if max_workers < 1:
                raise ValueError("max_workers must be >= 1")
            if max_workers != self.max_concurrent_queries:
                self._close_query_connections()
            self.max_concurrent_queries = max_workers

    def begin_episode(self, episode_idx: int) -> None:
        """Notify the client that a new episode is starting.

        Called by ``run_episode`` before the first inference of each episode.
        The default stores the index on ``self._eval_episode_idx``. Clients
        whose server keeps state across a persistent connection can use it to
        mark episode boundaries in their requests; subclasses that need to
        notify a server-side session should override and call
        ``super().begin_episode(episode_idx)``.
        """
        self._eval_episode_idx = episode_idx


    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        """Return ``{"action": np.ndarray, "viz": np.ndarray | None}``.

        Default flow: extract -> pack -> query -> unpack -> postprocess ->
        cache chunk -> step one action. Override entirely if your client needs
        a different control loop.

        When ``run_episode(render_policy_steps_only=True)`` skipped rendering
        for an observation this client does not query with, the camera groups
        of ``obs`` are unrendered placeholders; the step then only advances the
        cached chunk (no extraction, ``viz`` is None).
        """
        if not self._needs_refresh(env_id) and self._has_unrendered_groups(obs):
            return {"action": self._next_action(env_id), "viz": None}

        extracted = self._extract_observation(obs, env_id=env_id)

        if self._needs_refresh(env_id):
            request = self._pack_request(extracted, instruction)
            response = self._query_server(request)
            chunk = self._unpack_response(response)
            chunk = self._postprocess_chunk(chunk)
            self._set_chunk(env_id, chunk)

        action = self._next_action(env_id)
        viz = self._build_visualization(extracted)
        return {"action": action, "viz": viz}

    def infer_batch(
        self, obs: Any, instruction: str, *, env_ids: list[int]
    ) -> dict[int, dict]:
        """Infer actions for several envs in one call.

        Returns ``{env_id: {"action": ..., "viz": ...}}`` with one entry per
        requested env. When two or more envs need a new chunk and the query
        mode (see :meth:`configure_queries`) allows it, their queries go out
        together: one batched request, or concurrent per-env requests. Every
        env still gets the same extraction, request, chunk and action as with
        the serial loop over :meth:`infer`; otherwise this is that loop.
        """
        refresh_ids = [env_id for env_id in env_ids if self._needs_refresh(env_id)]
        if len(refresh_ids) < 2 or self._effective_query_mode() == "serial":
            return {
                env_id: self.infer(obs, instruction, env_id=env_id)
                for env_id in env_ids
            }

        # Same per-env work as ``infer``, split so that all queries are sent at once.
        skip_extraction = self._has_unrendered_groups(obs)
        refresh = set(refresh_ids)
        extracted = {
            env_id: self._extract_observation(obs, env_id=env_id)
            for env_id in env_ids
            if env_id in refresh or not skip_extraction
        }
        requests = [self._pack_request(extracted[env_id], instruction) for env_id in refresh_ids]
        responses = self._query_server_many(requests)
        for env_id, response in zip(refresh_ids, responses, strict=True):
            self._set_chunk(env_id, self._postprocess_chunk(self._unpack_response(response)))

        rets = {}
        for env_id in env_ids:
            action = self._next_action(env_id)
            viz = self._build_visualization(extracted[env_id]) if env_id in extracted else None
            rets[env_id] = {"action": action, "viz": viz}
        return rets

    def reset(self, *, env_id: int | None = None) -> None:
        """Clear per-episode state. ``env_id=None`` resets all envs.

        Subclasses with server-side session state should override to notify
        the server, then call ``super().reset(env_id=env_id)``.
        """
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
        else:
            self._chunks.pop(env_id, None)
            self._counters.pop(env_id, None)

    def close(self) -> None:
        """Release transport resources. Default: close the concurrent-query connections.

        Subclasses that override this should call ``super().close()``.
        """
        self._close_query_connections()

    def visualize(self, obs: Any, *, env_id: int = 0) -> np.ndarray | None:
        """Public convenience wrapper for callers that want the viz image."""
        return self._build_visualization(self._extract_observation(obs, env_id=env_id))

    def controls_until_observation(self, env_id: int) -> int:
        """Number of upcoming env steps whose observation ``infer`` will not query with.

        Called after ``infer`` for ``env_id``: 0 means the observation returned
        by the next ``env.step`` feeds a policy query; ``n`` means the next
        ``n`` observations only advance the cached chunk. ``run_episode`` uses
        it to render only consumed observations when
        ``render_policy_steps_only=True``. The default follows the chunking
        helpers below; a client that bypasses them reports 0 (every
        observation consumed). Clients that override ``infer`` with different
        refresh rules must override this consistently.
        """
        if self._needs_refresh(env_id):
            return 0
        return max(0, self.open_loop_horizon - self._counters[env_id])

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_observation(self, raw_obs: Any, *, env_id: int = 0) -> dict:
        """Convert the caller's native obs into a flat dict of numpy arrays.

        Repo-specific seam. Return whatever keys ``_pack_request`` expects;
        the contract between these two methods is owned by the subclass pair.
        """

    @abstractmethod
    def _pack_request(self, extracted_obs: dict, instruction: str) -> Any:
        """Build the server's wire-format request. Backend-specific."""

    @abstractmethod
    def _query_server(self, request: Any) -> Any:
        """Send the request and return the raw response. Transport-specific."""

    @abstractmethod
    def _unpack_response(self, response: Any) -> np.ndarray:
        """Return a ``(horizon, action_dim)`` numpy array."""

    # ------------------------------------------------------------------
    # Multi-env query hooks (optional; see configure_queries)
    # ------------------------------------------------------------------

    def _open_connection(self) -> Any:
        """Open one more server connection for concurrent queries.

        Required with ``supports_concurrent_queries``. Each connection has its
        own worker thread: this hook, ``_query_server_on`` and
        ``_close_connection`` for it run only on that thread, so transports
        whose sockets are not thread-safe (ZeroMQ) need no locks.
        """
        raise NotImplementedError

    def _query_server_on(self, connection: Any, request: Any) -> Any:
        """Like ``_query_server``, over a connection from ``_open_connection``.

        Required with ``supports_concurrent_queries``. Runs on the worker thread
        of ``connection``: touch only ``connection`` and ``request``, no other
        client state.
        """
        raise NotImplementedError

    def _close_connection(self, connection: Any) -> None:
        """Close a connection from ``_open_connection``. Default: its ``close()``, if any."""
        close = getattr(connection, "close", None)
        if close is not None:
            close()

    def _query_server_batch(self, requests: list[Any]) -> list[Any]:
        """Send several ``_pack_request`` requests in one server call.

        Required with ``supports_batched_queries``. Return one response per
        request, in request order, each accepted by ``_unpack_response``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Observation helpers (generic; no backend or camera knowledge)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_obs_term(raw_obs: Any, term_name: str) -> Any | None:
        """Look up an observation term by name across all observation groups.

        Scans every mapping-valued entry of ``raw_obs`` (observation groups
        like ``image_obs`` / ``viewport_cam`` / ``proprio_obs``) and returns
        the first batched value stored under ``term_name``, or None if no
        group has it. Term names are unique across groups in RoboLab
        registrations.
        """
        if not isinstance(raw_obs, dict):
            return None
        for group in raw_obs.values():
            getter = getattr(group, "get", None)
            if getter is None:
                continue
            value = getter(term_name)
            if value is not None:
                return value
        return None

    @staticmethod
    def _get_env_gt_state(raw_obs: Any, env_id: int) -> dict | None:
        """Return this env's ground-truth state snapshot, or None.

        ``run_episode`` attaches ``obs["gt_state"] = {env_id: state}`` when
        ``--enable-gt-state`` is set (schema: ``robolab/eval/gt_state.py``).
        """
        gt_state = raw_obs.get("gt_state") if isinstance(raw_obs, dict) else None
        if isinstance(gt_state, dict) and env_id in gt_state:
            return gt_state[env_id]
        return None

    @staticmethod
    def _to_numpy(value: Any, env_id: int = 0) -> np.ndarray:
        """Convert a batched tensor/array to NumPy, selecting one env's row."""
        if hasattr(value, "detach"):
            return value[env_id].detach().cpu().numpy()
        array = np.asarray(value)
        return array[env_id] if array.ndim > 0 else array

    @staticmethod
    def _has_unrendered_groups(raw_obs: Any) -> bool:
        """Whether an observation group of ``raw_obs`` was skipped because it was not rendered.

        Skipped groups are Isaac Lab ``UnrenderedObservationGroup`` placeholders
        (marked ``unrendered = True``); reading any of their terms raises.
        """
        return isinstance(raw_obs, dict) and any(
            getattr(group, "unrendered", False) is True for group in raw_obs.values()
        )

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Action post-processing (binarization, padding, sign flips).

        Default: identity.
        """
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray | None:
        """Image for logging/recording. Default: None."""
        return None

    # ------------------------------------------------------------------
    # Chunking helpers (usable or ignorable by subclasses)
    # ------------------------------------------------------------------

    def _needs_refresh(self, env_id: int) -> bool:
        return env_id not in self._chunks or self._counters[env_id] >= self.open_loop_horizon

    def _set_chunk(self, env_id: int, chunk: np.ndarray) -> None:
        self._chunks[env_id] = chunk
        self._counters[env_id] = 0

    def _next_action(self, env_id: int) -> np.ndarray:
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return action

    # ------------------------------------------------------------------
    # Multi-env query internals
    # ------------------------------------------------------------------

    def _effective_query_mode(self) -> str:
        if type(self).infer is not InferenceClient.infer:
            # A custom ``infer`` defines its own per-env flow; keep the loop over it.
            return "serial"
        mode = self.query_mode
        if mode == "auto":
            mode = "concurrent" if self.order_independent_server else "serial"
        if mode == "batch" and not self.supports_batched_queries:
            mode = "concurrent"
        if mode == "concurrent" and (not self.supports_concurrent_queries or self.max_concurrent_queries < 2):
            mode = "serial"
        return mode

    def _query_server_many(self, requests: list[Any]) -> list[Any]:
        """Responses for ``requests``, in order, sent as one batch or concurrently."""
        if self._effective_query_mode() == "batch":
            responses = list(self._query_server_batch(requests))
            if len(responses) != len(requests):
                raise RuntimeError(f"batched query returned {len(responses)} responses for {len(requests)} requests")
            return responses

        # Request i goes to connection i % n: a fixed assignment, results stay in request order.
        n = min(len(requests), self.max_concurrent_queries)
        self._open_query_connections(n)
        futures = [
            self._query_threads[worker].submit(
                lambda conn, reqs: [self._query_server_on(conn, req) for req in reqs],
                self._query_connections[worker],
                requests[worker::n],
            )
            for worker in range(n)
        ]
        # Wait for every worker before raising, so no thread still uses a connection.
        wait(futures)
        errors = [future.exception() for future in futures if future.exception() is not None]
        if errors:
            # A failed connection can hold a half-sent request; open new ones for the next call.
            self._close_query_connections(raise_errors=False)
            raise errors[0]
        responses = [None] * len(requests)
        for worker, future in enumerate(futures):
            responses[worker::n] = future.result()
        return responses

    def _open_query_connections(self, n: int) -> None:
        """Make sure that ``n`` connections exist, each opened on its own worker thread."""
        new_threads = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"policy-query-{index}")
            for index in range(len(self._query_threads), n)
        ]
        futures = [thread.submit(self._open_connection) for thread in new_threads]
        wait(futures)
        for thread, future in zip(new_threads, futures, strict=True):
            self._query_threads.append(thread)
            self._query_connections.append(future.result() if future.exception() is None else None)
        errors = [future.exception() for future in futures if future.exception() is not None]
        if errors:
            self._close_query_connections(raise_errors=False)
            raise errors[0]

    def _close_query_connections(self, *, raise_errors: bool = True) -> None:
        """Close every connection on its own thread, then stop the threads.

        With ``raise_errors=False`` (after a failed query) close errors are
        dropped, so that the query error is the one that the caller sees.
        """
        connections = getattr(self, "_query_connections", [])
        threads = getattr(self, "_query_threads", [])
        closes = [
            thread.submit(self._close_connection, connection)
            for thread, connection in zip(threads, connections, strict=True)
            if connection is not None
        ]
        for thread in threads:
            thread.shutdown(wait=True)
        connections.clear()
        threads.clear()
        errors = [close.exception() for close in closes if close.exception() is not None]
        if errors and raise_errors:
            raise errors[0]
