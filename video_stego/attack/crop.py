"""
attack/crop.py — 裁剪攻击模拟
Simulate a spatial cropping attack: remove a border region from every frame
and, optionally, resize back to the original dimensions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


class CropAttack:
    """Apply a spatial crop to a sequence of frames.

    Parameters
    ----------
    crop_pixels:
        Number of pixels to remove from each edge
        ``(top, bottom, left, right)``.  A single integer sets the same
        margin on all four sides.
    resize_back:
        If ``True`` the cropped frames are scaled back to the original
        ``(width, height)`` using bilinear interpolation.

    Examples
    --------
    >>> attack = CropAttack(crop_pixels=10, resize_back=True)
    >>> attacked = attack.apply(frames)
    """

    def __init__(
        self,
        crop_pixels: int | Tuple[int, int, int, int] = 10,
        resize_back: bool = True,
    ) -> None:
        if isinstance(crop_pixels, int):
            self.top = self.bottom = self.left = self.right = crop_pixels
        else:
            self.top, self.bottom, self.left, self.right = crop_pixels
        self.resize_back = resize_back

    def apply(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply the crop attack to *frames*.

        Parameters
        ----------
        frames:
            List of BGR ``uint8`` frames.

        Returns
        -------
        list of np.ndarray
            Cropped (and optionally resized) frames.
        """
        if not frames:
            return []
        orig_h, orig_w = frames[0].shape[:2]
        result = []
        for frame in frames:
            h, w = frame.shape[:2]
            y1 = self.top
            y2 = h - self.bottom if self.bottom else h
            x1 = self.left
            x2 = w - self.right if self.right else w
            cropped = frame[y1:y2, x1:x2]
            if self.resize_back:
                cropped = cv2.resize(cropped, (orig_w, orig_h))
            result.append(cropped)
        return result
