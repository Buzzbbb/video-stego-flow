"""
codec.py — 视频解码与编码模块
Video decoding and encoding utilities.

Provides :class:`VideoDecoder` for reading video frames and
:class:`VideoEncoder` for writing frames back to a video file.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


class VideoDecoder:
    """Decode a video file into individual BGR frames.

    Parameters
    ----------
    path:
        Path to the source video file.

    Examples
    --------
    >>> dec = VideoDecoder("input.mp4")
    >>> frames = dec.read_frames()
    >>> print(len(frames), dec.fps, dec.frame_size)
    """

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of the video."""
        return self.width, self.height

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames one by one without loading all into memory."""
        cap = cv2.VideoCapture(self.path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    def read_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
    ) -> List[np.ndarray]:
        """Read a slice of frames into a list.

        Parameters
        ----------
        start:
            Index of the first frame to include (0-based).
        end:
            Exclusive end index.  ``None`` means the last frame.
        step:
            Stride between selected frames.

        Returns
        -------
        list of np.ndarray
            Each array has shape ``(H, W, 3)`` with dtype ``uint8``.
        """
        cap = cv2.VideoCapture(self.path)
        frames: List[np.ndarray] = []
        idx = 0
        end_idx = end if end is not None else self.total_frames
        try:
            while idx < end_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx >= start and (idx - start) % step == 0:
                    frames.append(frame)
                idx += 1
        finally:
            cap.release()
        return frames


class VideoEncoder:
    """Write a sequence of BGR frames to a video file.

    Parameters
    ----------
    path:
        Output file path (e.g. ``"output.mp4"``).
    fps:
        Frames per second for the output video.
    frame_size:
        ``(width, height)`` tuple.
    fourcc:
        Four-character code for the codec.  Defaults to ``"mp4v"``.

    Examples
    --------
    >>> enc = VideoEncoder("output.mp4", fps=25, frame_size=(1280, 720))
    >>> enc.write_frames(frames)
    """

    def __init__(
        self,
        path: str,
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: str = "mp4v",
    ) -> None:
        self.path = path
        self.fps = fps
        self.frame_size = frame_size
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)

    def write_frames(self, frames: List[np.ndarray]) -> None:
        """Write *frames* to the output video file.

        Parameters
        ----------
        frames:
            Sequence of BGR ``uint8`` arrays with shape ``(H, W, 3)``.
        """
        writer = cv2.VideoWriter(
            self.path, self._fourcc, self.fps, self.frame_size
        )
        if not writer.isOpened():
            raise IOError(f"Cannot open video writer for: {self.path}")
        try:
            for frame in frames:
                if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                    frame = cv2.resize(frame, self.frame_size)
                writer.write(frame)
        finally:
            writer.release()
