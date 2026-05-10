"""
metrics.py — 客观指标计算模块
Objective quality and robustness metrics:

* :func:`compute_psnr`  — Peak Signal-to-Noise Ratio (dB)
* :func:`compute_ssim`  — Structural Similarity Index
* :func:`compute_ber`   — Bit Error Rate
* :func:`compute_video_psnr` — Average PSNR over a frame sequence
* :func:`compute_video_ssim` — Average SSIM over a frame sequence
"""

from __future__ import annotations

from typing import List

import numpy as np
from skimage.metrics import structural_similarity as _ssim


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute the PSNR between *original* and *processed* frames.

    Parameters
    ----------
    original, processed:
        BGR ``uint8`` arrays with identical shape.

    Returns
    -------
    float
        PSNR in dB.  Returns ``float('inf')`` if the frames are identical.
    """
    if original.shape != processed.shape:
        raise ValueError("Frames must have identical shape")
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    mse = np.mean((orig - proc) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute the SSIM between *original* and *processed* frames.

    Parameters
    ----------
    original, processed:
        BGR ``uint8`` arrays with identical shape.

    Returns
    -------
    float
        SSIM in the range [−1, 1].  Values near 1 indicate high similarity.
    """
    if original.shape != processed.shape:
        raise ValueError("Frames must have identical shape")
    # scikit-image expects channel_axis for colour images
    return float(
        _ssim(original, processed, channel_axis=2, data_range=255)
    )


def compute_ber(
    original_bits: np.ndarray, recovered_bits: np.ndarray
) -> float:
    """Compute the Bit Error Rate between two bit arrays.

    Parameters
    ----------
    original_bits, recovered_bits:
        1-D ``uint8`` arrays of equal length (values 0 or 1).

    Returns
    -------
    float
        Fraction of bits that differ, in the range [0, 1].
    """
    orig = np.asarray(original_bits, dtype=np.uint8)
    rec = np.asarray(recovered_bits, dtype=np.uint8)
    if orig.shape != rec.shape:
        raise ValueError("Bit arrays must have equal length")
    if len(orig) == 0:
        return 0.0
    return float(np.mean(orig != rec))


def compute_video_psnr(
    original_frames: List[np.ndarray],
    processed_frames: List[np.ndarray],
) -> float:
    """Average PSNR over a list of corresponding frame pairs.

    Parameters
    ----------
    original_frames, processed_frames:
        Lists of BGR ``uint8`` frames (must have the same length).

    Returns
    -------
    float
        Mean PSNR in dB across all frames.
    """
    if len(original_frames) != len(processed_frames):
        raise ValueError("Frame lists must have the same length")
    if not original_frames:
        return float("nan")
    values = [compute_psnr(o, p) for o, p in zip(original_frames, processed_frames)]
    finite = [v for v in values if v != float("inf")]
    if not finite:
        return float("inf")
    return float(np.mean(finite))


def compute_video_ssim(
    original_frames: List[np.ndarray],
    processed_frames: List[np.ndarray],
) -> float:
    """Average SSIM over a list of corresponding frame pairs.

    Parameters
    ----------
    original_frames, processed_frames:
        Lists of BGR ``uint8`` frames (must have the same length).

    Returns
    -------
    float
        Mean SSIM across all frames.
    """
    if len(original_frames) != len(processed_frames):
        raise ValueError("Frame lists must have the same length")
    if not original_frames:
        return float("nan")
    return float(
        np.mean([compute_ssim(o, p) for o, p in zip(original_frames, processed_frames)])
    )
