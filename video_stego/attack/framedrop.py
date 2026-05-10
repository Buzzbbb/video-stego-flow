"""
attack/framedrop.py — 抽帧攻击模拟
Simulate a frame-drop (temporal sub-sampling) attack: remove a subset of
frames from the sequence, optionally duplicating neighbours to maintain the
original count.
"""

from __future__ import annotations

from typing import List

import numpy as np


class FrameDropAttack:
    """Drop frames from a video sequence at a given rate.

    Parameters
    ----------
    drop_rate:
        Fraction of frames to discard, in the range ``(0, 1)``.
        E.g. ``0.1`` drops approximately every 10th frame.
    fill_gaps:
        If ``True``, each dropped frame is replaced by a copy of its
        immediately preceding neighbour so the output length matches the
        input.  If ``False``, the output is shorter than the input.
    seed:
        Optional random seed for reproducible frame selection.

    Examples
    --------
    >>> attack = FrameDropAttack(drop_rate=0.1, fill_gaps=True, seed=0)
    >>> attacked = attack.apply(frames)
    """

    def __init__(
        self,
        drop_rate: float = 0.1,
        fill_gaps: bool = True,
        seed: int | None = None,
    ) -> None:
        if not 0 < drop_rate < 1:
            raise ValueError("drop_rate must be in (0, 1)")
        self.drop_rate = drop_rate
        self.fill_gaps = fill_gaps
        self._rng = np.random.default_rng(seed)

    def apply(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply the frame-drop attack to *frames*.

        Parameters
        ----------
        frames:
            List of BGR ``uint8`` frames.

        Returns
        -------
        list of np.ndarray
            Frames after dropping.
        """
        if not frames:
            return []
        n = len(frames)
        n_drop = max(1, int(n * self.drop_rate))
        drop_indices = set(
            self._rng.choice(n, size=n_drop, replace=False).tolist()
        )
        result: List[np.ndarray] = []
        last_kept: np.ndarray = frames[0]
        for i, frame in enumerate(frames):
            if i in drop_indices:
                if self.fill_gaps:
                    result.append(last_kept.copy())
            else:
                last_kept = frame
                result.append(frame)
        return result
