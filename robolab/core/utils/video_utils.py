# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import logging
import queue
import threading
from collections.abc import Callable

import imageio.v2 as imageio
import numpy as np

logger = logging.getLogger(__name__)

_STOP = object()


class _DeferredFrame:
    __slots__ = ("make_frame",)

    def __init__(self, make_frame: Callable[[], np.ndarray]):
        self.make_frame = make_frame


class VideoWriter:
    """Streaming H.264 video writer.

    Pipes RGB frames to an ffmpeg subprocess (via imageio-ffmpeg) which encodes
    libx264/yuv420p with +faststart so the output plays in Chrome/Safari/Firefox.
    Memory is bounded by the encoder's lookahead, not by video length.

    Frames are handed to one background thread through a bounded FIFO queue, so
    the caller does not wait for ffmpeg to consume each frame. Frames are encoded
    in call order, every frame is encoded, and at most ``max_pending`` frames wait
    in the queue before :meth:`write` or :meth:`write_deferred` blocks. An error
    preparing or encoding a frame is raised from the next :meth:`write` /
    :meth:`write_deferred` or from :meth:`release`.
    """

    def __init__(self, video_path: str, fps: int, max_pending: int = 8):
        """Initialize video writer.

        Args:
            video_path: Path to output video file
            fps: Frames per second
            max_pending: Maximum number of frames queued for the encoder thread
        """
        self.video_path = video_path
        self.fps = fps
        self._writer = None
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        atexit.register(self.release)

    def write(self, frame: np.ndarray, *, take_ownership: bool = False):
        """Queue one RGB frame for encoding.

        Args:
            frame: (H, W, 3) uint8 frame.
            take_ownership: Set when the caller never modifies ``frame`` afterwards;
                otherwise the frame is copied before it is queued.
        """
        if frame is None:
            print(f"No frame to write to video writer; nothing written to file '{self.video_path}'")
            return
        self._raise_pending_error()
        self._start()
        self._queue.put(frame if take_ownership else np.array(frame, copy=True))

    def write_deferred(self, make_frame: Callable[[], np.ndarray]):
        """Queue a frame that the encoder thread produces by calling ``make_frame()``.

        The frame keeps its position in the call order relative to :meth:`write`.
        ``make_frame`` runs on the encoder thread, so it must only read data that
        nobody modifies afterwards; its result is encoded like a frame passed to
        ``write(..., take_ownership=True)``. An exception from ``make_frame`` is
        reported like an encoding error.
        """
        self._raise_pending_error()
        self._start()
        self._queue.put(_DeferredFrame(make_frame))

    def _start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._encode_frames, name="robolab-video-writer", daemon=True)
            self._thread.start()

    def _encode_frames(self):
        try:
            while True:
                frame = self._queue.get()
                if frame is _STOP:
                    return
                if isinstance(frame, _DeferredFrame):
                    frame = frame.make_frame()
                if self._writer is None:
                    # macro_block_size=2 pads odd-dimensioned frames so libx264+yuv420p (even-only) is safe.
                    # ultrafast/threads=1 keeps per-encoder CPU low so num_envs=10 doesn't oversubscribe.
                    self._writer = imageio.get_writer(
                        self.video_path,
                        fps=self.fps,
                        codec="libx264",
                        pixelformat="yuv420p",
                        macro_block_size=2,
                        ffmpeg_params=[
                            "-preset", "ultrafast",
                            "-threads", "1",
                            "-crf", "23",
                            "-movflags", "+faststart",
                        ],
                    )
                self._writer.append_data(frame)
        except BaseException as exc:  # surfaced to the producer by write()/release()
            self._error = exc
            # keep consuming so producers never block on a full queue
            while self._queue.get() is not _STOP:
                pass

    def _raise_pending_error(self):
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError(f"Video encoding failed for '{self.video_path}'") from error

    def release(self):
        """Encode all queued frames, then finalize the file.

        Raises:
            RuntimeError: If encoding a queued frame failed.
        """
        if self._thread is not None:
            self._queue.put(_STOP)
            self._thread.join()
            self._thread = None
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.exception("Failed to close video writer for '%s'", self.video_path)
            self._writer = None
        self._raise_pending_error()

    def __del__(self):
        try:
            self.release()
        except Exception:
            logger.exception("Video writer for '%s' failed", self.video_path)
