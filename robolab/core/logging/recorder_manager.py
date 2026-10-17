# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
from collections.abc import Sequence

import psutil
import torch
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg
from isaaclab.utils.datasets import EpisodeData

from robolab.core.logging.streaming_hdf5_handler import StreamingHDF5DatasetFileHandler

logger = logging.getLogger(__name__)


def _collect_tensor_leaves(key: str, value, leaves: list) -> bool:
    """Append ``(path, tensor)`` for every leaf of a nested recorder value; False if a leaf is not a tensor."""
    if isinstance(value, dict):
        return all(_collect_tensor_leaves(f"{key}/{sub_key}", sub_value, leaves) for sub_key, sub_value in value.items())
    if isinstance(value, torch.Tensor):
        leaves.append((key, value))
        return True
    return False


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_episode_data_size(episode: EpisodeData) -> dict:
    """Get size info about episode data buffer."""
    if episode.is_empty():
        return {"num_keys": 0, "total_elements": 0, "size_mb": 0.0}

    def count_elements(data, prefix=""):
        """Recursively count elements in nested dict of tensors."""
        info = {}
        total = 0
        for key, value in data.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                sub_info = count_elements(value, full_key)
                info.update(sub_info["details"])
                total += sub_info["total"]
            elif isinstance(value, torch.Tensor):
                num_elements = value.numel()
                size_mb = value.element_size() * num_elements / (1024 * 1024)
                info[full_key] = {"shape": list(value.shape), "elements": num_elements, "size_mb": size_mb}
                total += num_elements
        return {"details": info, "total": total}

    result = count_elements(episode.data)
    total_size_mb = sum(v.get("size_mb", 0) for v in result["details"].values())

    return {
        "num_keys": len(result["details"]),
        "total_elements": result["total"],
        "size_mb": total_size_mb,
        "details": result["details"]
    }


class RobolabRecorderManager(RecorderManager):
    """Custom recorder manager with streaming HDF5 support for memory-efficient long episodes.

    This manager allows:
    - Incremental flushing of data to disk during an episode (via flush_buffer())
    - Proper episode finalization without creating multiple demos
    - Memory-efficient recording for episodes with thousands of steps
    """

    initialized = False

    def __init__(self, cfg: RecorderManagerBaseCfg, env: ManagerBasedEnv):
        """Initialize the recorder manager with streaming HDF5 support."""
        self._term_names: list[str] = list()
        self._terms: dict = dict()

        # Track streaming state per environment
        self._streaming_active: dict[int, bool] = {}

        # Per-control recorder outputs not yet split into per-env EpisodeData (see _buffer_step_rows)
        self._pending_rows: dict[str, list[torch.Tensor]] = {}
        self._pending_env_ids: tuple[int, ...] = ()

        # Do nothing if cfg is None or an empty dict
        if not cfg:
            return

        # Call grandparent's __init__ to skip RecorderManager's file handler setup
        from isaaclab.managers.manager_base import ManagerBase
        ManagerBase.__init__(self, cfg, env)

        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if not isinstance(cfg, RecorderManagerBaseCfg):
            raise TypeError("Configuration for the recorder manager is not of type RecorderManagerBaseCfg.")

        # create episode data buffer indexed by environment id
        self._episodes = dict()
        for env_id in range(env.num_envs):
            self._episodes[env_id] = EpisodeData()
            self._streaming_active[env_id] = False

        self._env_name = getattr(env.cfg, "env_name", None)

        # HDF5 file handlers — created lazily on first write or when set_hdf5_file() is called.
        # This avoids creating empty data.hdf5 files that are never written to.
        self._dataset_file_handler = None
        self._failed_episode_dataset_file_handler = None
        self._hdf5_initialized = False

        self._exported_successful_episode_count = {}
        self._exported_failed_episode_count = {}

        # Track current episode index per environment (for overwriting existing episodes)
        self._current_episode_index: dict[int, int | None] = {}
        for env_id in range(env.num_envs):
            self._current_episode_index[env_id] = None

        # Auto-flush configuration
        self._flush_interval: int = 500  # Flush every N steps (0 = disabled)
        self._step_count: dict[int, int] = {}
        for env_id in range(env.num_envs):
            self._step_count[env_id] = 0

        self._auto_flush_verbose: bool = False  # Print diagnostics on auto-flush

        self.initialized = True

    @property
    def _episodes(self) -> dict[int, EpisodeData]:
        """Per-env episode buffers, including every recorded control.

        Step rows buffered by :meth:`_buffer_step_rows` are moved into the buffers before
        any access, so every reader and writer (flush, export, reset callbacks,
        :meth:`get_episode`) sees exactly the data per-step accumulation would have built.
        """
        if self._pending_rows:
            self._materialize_pending_rows()
        return self._episode_store

    @_episodes.setter
    def _episodes(self, episodes: dict[int, EpisodeData]) -> None:
        self._episode_store = episodes

    def _buffer_step_rows(self, key: str, value, env_ids: Sequence[int]) -> None:
        """Record one control's full-batch term output for ``env_ids``.

        Instead of one ``EpisodeData.add`` per env and leaf per control, each tensor leaf is
        snapshotted once per control (sources may be reused simulator buffers). The per-env
        rows are split off and appended in one operation per env and leaf when the episode
        buffers are next accessed, or when the recorded env set changes. Values, dtypes,
        shapes and row order are those of :meth:`_add_full_batch_to_episodes`.
        """
        leaves: list[tuple[str, torch.Tensor]] = []
        if not _collect_tensor_leaves(key, value, leaves):
            self._add_full_batch_to_episodes(key, value, env_ids)
            return
        env_ids = tuple(env_ids)
        if self._pending_rows and env_ids != self._pending_env_ids:
            self._materialize_pending_rows()
        self._pending_env_ids = env_ids
        for leaf_key, leaf in leaves:
            self._pending_rows.setdefault(leaf_key, []).append(leaf.clone())

    def _materialize_pending_rows(self) -> None:
        """Append buffered step rows to the per-env episode buffers, in recording order."""
        pending, self._pending_rows = self._pending_rows, {}
        env_ids = self._pending_env_ids
        # same row selection as _add_full_batch_to_episodes
        select_rows = len(env_ids) < self._env.num_envs
        episodes = self._episode_store
        for key, snapshots in pending.items():
            steps = torch.stack(snapshots)  # (controls, num_envs, ...)
            *groups, leaf_name = key.split("/")
            for value_index, env_id in enumerate(env_ids):
                episode = episodes.get(env_id)
                if episode is None:
                    episode = episodes[env_id] = EpisodeData()
                    episode.env_id = env_id
                rows = steps[:, env_id if select_rows else value_index]
                node = episode.data
                for group in groups:
                    node = node.setdefault(group, {})
                if leaf_name in node:
                    node[leaf_name] = torch.cat((node[leaf_name], rows))
                else:
                    node[leaf_name] = rows.clone(memory_format=torch.contiguous_format)

    def _add_full_batch_to_episodes(self, key: str, value, env_ids: Sequence[int]) -> None:
        """Record selected full-batch rows without allocating an intermediate gather.

        EpisodeData.add owns the snapshot copy. Scalar row selection also works
        for uint16/uint32/uint64 on CPU and CUDA, unlike advanced indexing.
        Reset callbacks still use upstream add_to_episodes: their values are
        already subset-shaped, not indexed by global environment ID.
        """
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self._add_full_batch_to_episodes(f"{key}/{sub_key}", sub_value, env_ids)
            return
        # Match the existing positional contract for non-tensor leaves and
        # full-batch exports; only subset tensor payloads need global row IDs.
        select_rows = isinstance(value, torch.Tensor) and len(env_ids) < self._env.num_envs
        for value_index, env_id in enumerate(env_ids):
            if env_id not in self._episodes:
                self._episodes[env_id] = EpisodeData()
                self._episodes[env_id].env_id = env_id
            self._episodes[env_id].add(key, value[env_id if select_rows else value_index])

    def set_success_to_episodes(self, env_ids: Sequence[int] | None, success_values: torch.Tensor):
        """Transfer success flags together rather than synchronizing once per env."""
        if not self.active_terms:
            return
        if env_ids is None:
            env_ids = range(self._env.num_envs)
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        for env_id, success in zip(env_ids, success_values.reshape(-1).tolist()):
            self._episodes[env_id].success = success

    def get_term(self, term_type: type) -> object | None:
        """Return the first active recorder term of the given type, or None.

        Public accessor so callers (e.g. the GT-state exporter) don't have to
        reach into the private ``_terms`` dict.
        """
        for term in self._terms.values():
            if isinstance(term, term_type):
                return term
        return None

    def set_flush_interval(self, interval: int, verbose: bool = False):
        """Set the automatic flush interval.

        Args:
            interval: Number of steps between automatic flushes. Set to 0 to disable auto-flush.
            verbose: If True, print diagnostics when auto-flushing.
        """
        self._flush_interval = interval
        self._auto_flush_verbose = verbose
        if interval > 0:
            print(f"[RobolabRecorderManager] Auto-flush enabled: every {interval} steps")
        else:
            print(f"[RobolabRecorderManager] Auto-flush disabled")

    def _ensure_hdf5_handler(self, filename: str = None):
        """Create or switch the HDF5 file handler.

        Called lazily on first write or explicitly by set_hdf5_file().
        """
        cfg: RecorderManagerBaseCfg = self.cfg
        if cfg.dataset_export_mode == DatasetExportMode.EXPORT_NONE:
            return

        if filename is None:
            filename = cfg.dataset_filename

        # Close existing handler if switching files
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.close()
        else:
            self._dataset_file_handler = StreamingHDF5DatasetFileHandler()

        filepath = os.path.join(cfg.dataset_export_dir_path, filename)
        self._dataset_file_handler.create(filepath, env_name=self._env_name)

        if cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
            if self._failed_episode_dataset_file_handler is not None:
                self._failed_episode_dataset_file_handler.close()
            else:
                self._failed_episode_dataset_file_handler = StreamingHDF5DatasetFileHandler()
            failed_path = os.path.join(cfg.dataset_export_dir_path, f"{filename}_failed")
            self._failed_episode_dataset_file_handler.create(failed_path, env_name=self._env_name)

        self._hdf5_initialized = True

    def set_hdf5_file(self, filename: str):
        """Switch to a new HDF5 file. Closes the current file and opens/creates the new one.

        Args:
            filename: New filename (e.g. "run_0.hdf5"). Written to the same export directory.
        """
        self._ensure_hdf5_handler(filename)

    def set_episode_index(self, episode_index: int, env_ids: Sequence[int] | None = None):
        """Set the episode index to use for the next flush/export.

        If the episode already exists in the HDF5 file, it will be deleted and recreated.

        Args:
            episode_index: The episode index (demo_N) to use.
            env_ids: The environment ids. Defaults to None (all environments).
        """
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        for env_id in env_ids:
            self._current_episode_index[env_id] = episode_index

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the recorder data, but does not clear the buffer.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """

        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return {}

        # resolve environment ids
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))

        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        info = {}
        if hasattr(self, "_terms"):
            for term in self._terms.values():
                data = term.reset(env_ids=env_ids)
                info[term.__class__.__name__] = data

        # Disable clearing buffer - we handle this manually via clear()
        # for env_id in env_ids:
        #     self._episodes[env_id] = EpisodeData()

        return info

    def _active_env_ids(self) -> Sequence[int]:
        """Env ids that are still actively recording.

        Frozen envs have already had their demo finalized via export_episodes;
        further recorder writes for them must be suppressed to prevent two failure
        modes:
        - Within-run corruption: auto-flush with episode_index=None falls through
          to _demo_count in StreamingHDF5DatasetFileHandler.append_data, leaking
          data into whichever env's demo is currently open at that index.
        - Cross-run contamination: leftover _episodes[N] persists across runs
          (recorder state is not reset between run_episode calls in the eval flow)
          and would land in run K+1's demo_N on the first auto-flush.

        Falls back to all envs if the env doesn't expose ``_frozen_envs`` — matches
        upstream RecorderManager behavior for non-RobolabEnv subclasses.
        """
        active_ids = getattr(self._env, "_active_env_ids", None)
        if active_ids is not None:
            return active_ids
        frozen = getattr(self._env, "_frozen_envs", None)
        if frozen is None:
            return list(range(self._env.num_envs))
        return (~frozen).nonzero(as_tuple=False).squeeze(-1).tolist()

    def record_pre_step(self) -> None:
        """Trigger recorder terms for pre-step functions, skipping frozen envs.

        See _active_env_ids for rationale.
        """
        if len(self.active_terms) == 0:
            return

        active = self._active_env_ids()
        if not active:
            return

        for term in self._terms.values():
            key, value = term.record_pre_step()
            if key is None:
                continue
            self._buffer_step_rows(key, value, active)

    def record_post_step(self) -> None:
        """Trigger recorder terms for post-step functions with auto-flush support.

        Skips frozen envs; see _active_env_ids.
        """
        if len(self.active_terms) == 0:
            return

        active = self._active_env_ids()
        if not active:
            return

        for term in self._terms.values():
            key, value = term.record_post_step()
            if key is None:
                continue
            self._buffer_step_rows(key, value, active)

        if self._flush_interval > 0:
            for env_id in active:
                self._step_count[env_id] += 1

                if self._step_count[env_id] >= self._flush_interval:
                    self.flush_buffer(env_ids=[env_id], verbose=self._auto_flush_verbose)
                    self._step_count[env_id] = 0

    def record_pre_reset(
        self,
        env_ids: Sequence[int] | None,
        force_export_or_skip=None,
    ) -> None:
        """Filter env_ids to non-frozen, then delegate to upstream.

        ManagerBasedRLEnv.step() passes the unfiltered reset_buf, which keeps
        flagging frozen envs every step (their timeout/success conditions stay
        True after termination). Without this filter, upstream record_pre_reset
        would re-record finalized envs, feeding the cross-run leak.
        """
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        active_set = set(self._active_env_ids())
        env_ids = [e for e in env_ids if e in active_set]
        if not env_ids:
            return

        super().record_pre_reset(env_ids, force_export_or_skip=force_export_or_skip)

        # Preserve upstream's optional export ordering, then override the
        # in-memory success flag with the OR of all non-timeout terms. This
        # supports success_<obj> tasks when export follows in env._reset_idx.
        if hasattr(self._env, "termination_manager"):
            success = self._env.termination_manager.terminated[env_ids]
            self.set_success_to_episodes(env_ids, success)

    def record_post_reset(self, env_ids: Sequence[int] | None) -> None:
        """Filter env_ids to non-frozen, then delegate to upstream.

        See record_pre_reset for rationale.
        """
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        active_set = set(self._active_env_ids())
        env_ids = [e for e in env_ids if e in active_set]
        if not env_ids:
            return

        super().record_post_reset(env_ids)

    def flush_buffer(self, env_ids: Sequence[int] | None = None, verbose: bool = True):
        """Flush the current buffer to disk without finalizing the episode.

        This writes the accumulated data to the HDF5 file and clears the in-memory
        buffer, but keeps the episode "open" so subsequent data gets appended to
        the same demo_N entry.

        Use this periodically during long episodes to prevent OOM errors.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
            verbose: If True, print diagnostic information about memory and HDF5 state.
        """
        # Do nothing if no active recorder terms
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # Lazy-init HDF5 on first write
        if not self._hdf5_initialized:
            self._ensure_hdf5_handler()

        for env_id in env_ids:
            if env_id not in self._episodes or self._episodes[env_id].is_empty():
                continue

            # Get target file handler based on export mode
            # Note: During streaming, we don't know success yet, so we use the main handler
            target_handler = self._dataset_file_handler
            if target_handler is None:
                continue

            # === BEFORE FLUSH DIAGNOSTICS ===
            if verbose:
                ram_before = get_memory_usage_mb()
                buffer_info = get_episode_data_size(self._episodes[env_id])
                hdf5_before = target_handler.get_episode_stats() if hasattr(target_handler, 'get_episode_stats') else {}

                print("\n" + "="*60)
                print(f"[FLUSH_BUFFER] env_id={env_id} - BEFORE FLUSH")
                print("="*60)
                print(f"  RAM Usage: {ram_before:.1f} MB")
                print(f"  In-Memory Buffer: {buffer_info['num_keys']} keys, {buffer_info['total_elements']:,} elements, {buffer_info['size_mb']:.2f} MB")
                if hdf5_before:
                    print(f"  HDF5: demo_{hdf5_before.get('demo_count', 0)}, {hdf5_before.get('current_samples', 0):,} samples")

            # Get episode index for this environment (if set)
            episode_index = self._current_episode_index.get(env_id)

            # Append data to the current episode (creates one if not started)
            # If episode_index is set and episode exists, it will be deleted and recreated
            target_handler.append_data(self._episodes[env_id], episode_index=episode_index)
            self._streaming_active[env_id] = True

            # Clear the in-memory buffer (but episode stays open in HDF5)
            self._episodes[env_id] = EpisodeData()

            # Force garbage collection to actually free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # === AFTER FLUSH DIAGNOSTICS ===
            if verbose:
                ram_after = get_memory_usage_mb()
                hdf5_after = target_handler.get_episode_stats() if hasattr(target_handler, 'get_episode_stats') else {}
                samples_added = hdf5_after.get('current_samples', 0) - hdf5_before.get('current_samples', 0)

                print(f"[FLUSH_BUFFER] AFTER: RAM {ram_after:.1f} MB (freed {ram_before - ram_after:.1f} MB), "
                      f"HDF5 demo_{hdf5_after.get('demo_count', 0)} now has {hdf5_after.get('current_samples', 0):,} samples (+{samples_added})")
                print("="*60 + "\n")

        # Clear any term buffers if they support it
        if hasattr(self, "_terms"):
            for term in self._terms.values():
                if hasattr(term, "clear"):
                    term.clear()

    def export_episodes(self, env_ids: Sequence[int] | None = None) -> None:
        """Concludes and exports the episodes for the given environment ids.

        If streaming is active, this finalizes the current demo. Otherwise,
        it writes the complete episode as a new demo (standard behavior).

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # Filter to envs with a pending episode to finalize. An env with no
        # episode_index and no active streaming session has either never been
        # armed via set_episode_index or was already exported (we reset both to
        # the no-pending state at finalize). Proceeding for such envs would
        # spuriously write an extra demo via the write_episode fallback after
        # record_final_status adds an entry to its (otherwise-empty) buffer.
        env_ids = [
            e for e in env_ids
            if self._current_episode_index.get(e) is not None
            or self._streaming_active.get(e, False)
        ]
        if not env_ids:
            return

        # Lazy-init HDF5 on first write
        if not self._hdf5_initialized:
            self._ensure_hdf5_handler()

        # Final-status terms return full batches; append only the exported rows.
        for term in self._terms.values():
            if hasattr(term, 'record_final_status'):
                key, value = term.record_final_status()
                if key is not None and value is not None:
                    self._add_full_batch_to_episodes(key, value, env_ids)

        # Get export mode from config
        cfg: RecorderManagerBaseCfg = self.cfg  # type: ignore

        for env_id in env_ids:
            episode = self._episodes.get(env_id)
            is_streaming = self._streaming_active.get(env_id, False)

            # Determine target handler based on success (for non-streaming final export)
            episode_succeeded = episode.success if episode else None
            target_handler = None

            if (cfg.dataset_export_mode == DatasetExportMode.EXPORT_ALL) or (
                cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_ONLY and episode_succeeded
            ):
                target_handler = self._dataset_file_handler
            elif cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
                if episode_succeeded:
                    target_handler = self._dataset_file_handler
                else:
                    target_handler = self._failed_episode_dataset_file_handler

            if target_handler is None:
                continue

            # Get episode index for this environment (if set)
            episode_index = self._current_episode_index.get(env_id)

            if is_streaming:
                # Streaming mode: append any remaining data and finalize
                if episode and not episode.is_empty():
                    target_handler.append_data(episode, episode_index=episode_index)
                target_handler.end_episode(success=episode_succeeded, episode_index=episode_index)
                self._streaming_active[env_id] = False
            else:
                # Standard mode: write complete episode (with optional episode_index)
                if episode and not episode.is_empty():
                    if episode_index is not None:
                        target_handler.begin_episode(episode.seed, episode_index=episode_index)
                        target_handler.append_data(episode, episode_index=episode_index)
                        target_handler.end_episode(success=episode_succeeded, episode_index=episode_index)
                    else:
                        target_handler.write_episode(episode)

            # Clear in-memory buffer and reset auto-flush counter now that
            # this env's demo is finalized in HDF5. Pairs with the frozen-env
            # skip in record_post_step: belt for the data we'll never record
            # again, suspenders for the data already in the buffer at export.
            self._episodes[env_id] = EpisodeData()
            self._step_count[env_id] = 0

            # Reset episode index after export
            self._current_episode_index[env_id] = None

            # Update counts
            if episode_succeeded:
                if env_id not in self._exported_successful_episode_count:
                    self._exported_successful_episode_count[env_id] = 0
                self._exported_successful_episode_count[env_id] += 1
            else:
                if env_id not in self._exported_failed_episode_count:
                    self._exported_failed_episode_count[env_id] = 0
                self._exported_failed_episode_count[env_id] += 1

    def clear(self, env_ids: Sequence[int] | None = None):
        """Clear the buffer of the recorder manager.

        Note: If streaming is active, calling clear() without export_episodes()
        will leave an incomplete episode in the HDF5 file.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))

        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        if hasattr(self, "_episodes"):
            for env_id in env_ids:
                # If a streaming episode is open in the file handler, finalize
                # it before discarding the in-memory buffer; otherwise the
                # demo group is left half-finalized on disk with no `success`.
                if self._streaming_active.get(env_id, False):
                    episode_index = (
                        self._current_episode_index.get(env_id)
                        if hasattr(self, "_current_episode_index") else None
                    )
                    for handler in (
                        getattr(self, "_dataset_file_handler", None),
                        getattr(self, "_failed_episode_dataset_file_handler", None),
                    ):
                        if handler is not None and hasattr(handler, "end_episode"):
                            try:
                                handler.end_episode(episode_index=episode_index)
                            except Exception:
                                logger.exception(
                                    "Failed to finalize streaming episode for "
                                    "env_id=%d during clear(); demo may be incomplete.",
                                    env_id,
                                )
                self._episodes[env_id] = EpisodeData()
                # Reset streaming state since we're clearing
                self._streaming_active[env_id] = False
                # Reset episode index
                if hasattr(self, "_current_episode_index"):
                    self._current_episode_index[env_id] = None
                # Reset step count for auto-flush
                if hasattr(self, "_step_count"):
                    self._step_count[env_id] = 0

        if hasattr(self, "_terms"):
            # Call clear on terms, if provided
            for term in self._terms.values():
                if hasattr(term, "clear"):
                    term.clear()

        return {}
