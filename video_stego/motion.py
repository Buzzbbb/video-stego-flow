"""
motion.py — 运动区域分析模块
Motion region analysis for steganographic embedding position selection.

:class:`MotionAnalyzer` computes inter-frame motion and returns binary or
scored region masks that guide where to embed secret data.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


class MotionAnalyzer:
    """Analyse motion between consecutive frames.

    Parameters
    ----------
    block_size:
        Side length (in pixels) of the non-overlapping blocks used for
        block-level motion scoring.
    motion_threshold:
        Mean absolute pixel difference below which a block is considered
        *static*.  Blocks exceeding this value are classified as *moving*.

    Examples
    --------
    >>> analyzer = MotionAnalyzer(block_size=16, motion_threshold=10.0)
    >>> mask = analyzer.motion_mask(frame_a, frame_b)
    """

    def __init__(
        self,
        block_size: int = 16,
        motion_threshold: float = 10.0,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size
        self.motion_threshold = motion_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def frame_difference(
        self, frame_a: np.ndarray, frame_b: np.ndarray
    ) -> np.ndarray:
        """Return the per-pixel absolute difference between two BGR frames.

        Parameters
        ----------
        frame_a, frame_b:
            BGR ``uint8`` arrays with identical shape.

        Returns
        -------
        np.ndarray
            Grayscale (single-channel) absolute difference map, ``uint8``.
        """
        self._check_shape(frame_a, frame_b)
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float64)
        diff = np.abs(gray_a - gray_b)
        return np.clip(diff, 0, 255).astype(np.uint8)

    def motion_mask(
        self, frame_a: np.ndarray, frame_b: np.ndarray
    ) -> np.ndarray:
        """Return a binary mask where *True* indicates moving blocks.

        The mask has the same spatial shape as the input frames (``H × W``).
        Each block of size :attr:`block_size` × :attr:`block_size` is set to
        ``True`` if the mean absolute difference within the block exceeds
        :attr:`motion_threshold`.

        Parameters
        ----------
        frame_a, frame_b:
            BGR ``uint8`` arrays with identical shape.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(H, W)``.
        """
        diff = self.frame_difference(frame_a, frame_b).astype(np.float64)
        h, w = diff.shape
        mask = np.zeros((h, w), dtype=bool)
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                block = diff[y:y + self.block_size, x:x + self.block_size]
                if block.mean() > self.motion_threshold:
                    mask[y:y + self.block_size, x:x + self.block_size] = True
        return mask

    def motion_score_map(
        self, frame_a: np.ndarray, frame_b: np.ndarray
    ) -> np.ndarray:
        """Return a float score map of block-level motion intensity.

        Each block is assigned the mean absolute difference of its pixels.
        The resulting map has the same spatial dimensions as the input frames.

        Returns
        -------
        np.ndarray
            Float64 array of shape ``(H, W)`` with per-pixel motion scores.
        """
        diff = self.frame_difference(frame_a, frame_b).astype(np.float64)
        h, w = diff.shape
        score_map = np.zeros((h, w), dtype=np.float64)
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                block = diff[y:y + self.block_size, x:x + self.block_size]
                score_map[y:y + self.block_size, x:x + self.block_size] = block.mean()
        return score_map

    def select_embedding_regions(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        top_k_blocks: Optional[int] = None,
    ) -> List[List[Tuple[int, int]]]:
        """Select the best embedding block positions for each selected frame.

        For each consecutive pair ``(frames[i-1], frames[i])`` the method
        ranks blocks by their motion score and returns the top-*k* block
        origins (upper-left corners in ``(row, col)`` order).

        Parameters
        ----------
        frames:
            Complete list of decoded frames (BGR ``uint8``).
        frame_indices:
            Indices (into *frames*) of the frames selected for embedding.
        top_k_blocks:
            Maximum number of block positions to return per frame.
            ``None`` returns all blocks sorted by score (highest first).

        Returns
        -------
        list of list of (row, col) tuples
            One inner list per frame index, giving the selected block origins.
        """
        h, w = frames[0].shape[:2]
        block_origins = [
            (y, x)
            for y in range(0, h, self.block_size)
            for x in range(0, w, self.block_size)
        ]

        result = []
        for idx in frame_indices:
            prev_idx = max(0, idx - 1)
            score_map = self.motion_score_map(frames[prev_idx], frames[idx])
            scores = np.array(
                [score_map[r, c] for r, c in block_origins], dtype=np.float64
            )
            order = np.argsort(scores)[::-1]
            sorted_origins = [block_origins[i] for i in order]
            if top_k_blocks is not None:
                sorted_origins = sorted_origins[:top_k_blocks]
            result.append(sorted_origins)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_shape(a: np.ndarray, b: np.ndarray) -> None:
        if a.shape != b.shape:
            raise ValueError(
                f"Frames must have the same shape; got {a.shape} and {b.shape}"
            )
