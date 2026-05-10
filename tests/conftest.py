"""
Shared test fixtures and helpers.
"""
import numpy as np
import pytest


def make_frame(height: int = 64, width: int = 64, seed: int = 0) -> np.ndarray:
    """Return a random BGR uint8 frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def make_static_frame_pair(height: int = 64, width: int = 64, seed: int = 0):
    """Return two nearly-identical BGR frames (very small difference)."""
    frame_a = make_frame(height, width, seed)
    frame_b = frame_a.copy()
    # Tiny noise so they are not bit-identical but still look static
    rng = np.random.default_rng(seed + 1)
    noise = rng.integers(-1, 2, size=(height, width, 3)).astype(np.int16)
    frame_b = np.clip(frame_b.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame_a, frame_b
