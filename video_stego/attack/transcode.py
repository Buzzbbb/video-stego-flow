"""
attack/transcode.py — 转码攻击模拟
Simulate re-encoding (transcoding) by writing frames to a video and reading
them back with potentially different codec settings.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import cv2
import numpy as np

from ..codec import VideoEncoder, VideoDecoder


class TranscodeAttack:
    """Simulate a transcoding attack by re-encoding frames via an on-disk video.

    The stego frames are written to a temporary video file using the specified
    *fourcc* codec and *quality* hint, then decoded back.  Lossy codecs (e.g.
    ``"MJPG"``) will introduce JPEG-like compression artefacts that degrade
    hidden data embedded with fragile methods such as LSB.

    Parameters
    ----------
    fourcc:
        FourCC codec identifier (e.g. ``"mp4v"``, ``"MJPG"``).
    quality:
        VideoWriter quality parameter passed to OpenCV (0-100).  Only
        effective for codecs that respect it (e.g. ``"MJPG"``).

    Examples
    --------
    >>> attack = TranscodeAttack(fourcc="MJPG", quality=70)
    >>> attacked = attack.apply(frames, fps=25.0)
    """

    def __init__(self, fourcc: str = "mp4v", quality: int = 95) -> None:
        self.fourcc = fourcc
        self.quality = quality

    def apply(
        self, frames: List[np.ndarray], fps: float = 25.0
    ) -> List[np.ndarray]:
        """Apply the transcoding attack to *frames*.

        Parameters
        ----------
        frames:
            List of BGR ``uint8`` frames (must all have the same shape).
        fps:
            Frame rate to use for the temporary video file.

        Returns
        -------
        list of np.ndarray
            Decoded frames after transcoding.
        """
        if not frames:
            return []
        h, w = frames[0].shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
            tmp_path = f.name
        try:
            enc = VideoEncoder(tmp_path, fps=fps, frame_size=(w, h), fourcc=self.fourcc)
            enc.write_frames(frames)
            dec = VideoDecoder(tmp_path)
            return dec.read_frames()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
