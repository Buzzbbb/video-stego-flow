"""
attack/compress.py — 压缩攻击模拟
Simulate a JPEG-like compression attack by re-encoding each frame as a JPEG
at a reduced quality level and decoding it back to a numpy array.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


class CompressionAttack:
    """Apply JPEG compression to every frame in a sequence.

    Parameters
    ----------
    quality:
        JPEG quality factor (1–100).  Lower values introduce more artefacts.

    Examples
    --------
    >>> attack = CompressionAttack(quality=50)
    >>> attacked = attack.apply(frames)
    """

    def __init__(self, quality: int = 75) -> None:
        if not 1 <= quality <= 100:
            raise ValueError("quality must be between 1 and 100")
        self.quality = quality

    def apply(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply JPEG compression to *frames*.

        Each frame is encoded to an in-memory JPEG buffer and then decoded
        back, introducing lossy compression artefacts.

        Parameters
        ----------
        frames:
            List of BGR ``uint8`` frames.

        Returns
        -------
        list of np.ndarray
            Compressed-then-decompressed frames with the same shape as inputs.
        """
        if not frames:
            return []
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        result: List[np.ndarray] = []
        for frame in frames:
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            result.append(decoded)
        return result
