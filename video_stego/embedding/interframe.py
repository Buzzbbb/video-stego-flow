"""
embedding/interframe.py — 帧间冗余区域嵌入模块
Inter-frame redundancy steganography: embed bits in the temporal difference
between consecutive frames, targeting regions classified as redundant
(low-motion) so that modifications blend into static background.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..message import MessageFragmenter
from ..motion import MotionAnalyzer


class InterFrameEmbedder:
    """Embed and extract a secret message using inter-frame redundancy.

    The embedder works on pairs of consecutive frames.  Blocks where motion is
    *below* the :attr:`~MotionAnalyzer.motion_threshold` are considered
    *redundant* — they are nearly identical in adjacent frames — and are used
    as embedding hosts.

    Within each redundant block the average Y-channel value is adjusted by
    ±``delta`` so that its parity encodes the target bit:

    * bit = 0 → mean Y value rounded to the nearest even integer
    * bit = 1 → mean Y value rounded to the nearest odd integer

    Parameters
    ----------
    delta:
        Magnitude of the embedding perturbation (in luminance units).
    block_size:
        Side length (pixels) of the non-overlapping blocks used for embedding.
    motion_threshold:
        Mean absolute pixel difference above which a block is considered
        *moving* and therefore excluded from embedding.

    Examples
    --------
    >>> emb = InterFrameEmbedder(delta=3, block_size=8)
    >>> stego = emb.embed(frame, prev_frame, bits)
    >>> recovered = emb.extract(stego, len(bits))
    """

    def __init__(
        self,
        delta: int = 3,
        block_size: int = 8,
        motion_threshold: float = 10.0,
    ) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = delta
        self.block_size = block_size
        self._analyzer = MotionAnalyzer(
            block_size=block_size, motion_threshold=motion_threshold
        )

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    def capacity_bits(
        self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of embeddable bits.

        If *prev_frame* is supplied, only static (non-moving) blocks are
        counted.  Otherwise, the total number of blocks is returned as a
        conservative upper bound.
        """
        h, w = frame.shape[:2]
        total_blocks = (h // self.block_size) * (w // self.block_size)
        if prev_frame is None:
            return total_blocks
        motion_mask = self._analyzer.motion_mask(prev_frame, frame)
        static_blocks = 0
        for r in range(0, h, self.block_size):
            for c in range(0, w, self.block_size):
                block_mask = motion_mask[r : r + self.block_size, c : c + self.block_size]
                if not block_mask.any():
                    static_blocks += 1
        return static_blocks

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        bits: np.ndarray,
    ) -> np.ndarray:
        """Embed *bits* into the redundant (static) regions of *frame*.

        Parameters
        ----------
        frame:
            The cover frame to carry the hidden data (BGR ``uint8``).
        prev_frame:
            The preceding frame, used to identify redundant regions.
        bits:
            1-D ``uint8`` bit array to embed.

        Returns
        -------
        np.ndarray
            Stego BGR ``uint8`` frame (same shape as *frame*).
        """
        motion_mask = self._analyzer.motion_mask(prev_frame, frame)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float64)

        h, w = y.shape
        bits = np.asarray(bits, dtype=np.uint8)
        bit_idx = 0

        for r in range(0, h - self.block_size + 1, self.block_size):
            for c in range(0, w - self.block_size + 1, self.block_size):
                if bit_idx >= len(bits):
                    break
                # Embed only in static (non-moving) blocks
                block_mask = motion_mask[r : r + self.block_size, c : c + self.block_size]
                if block_mask.any():
                    continue
                block = y[r : r + self.block_size, c : c + self.block_size]
                mean_val = block.mean()
                target_bit = int(bits[bit_idx])
                mean_val = self._adjust_mean(mean_val, target_bit)
                # Shift the entire block by the difference
                shift = mean_val - block.mean()
                y[r : r + self.block_size, c : c + self.block_size] = np.clip(
                    block + shift, 0, 255
                )
                bit_idx += 1

        if bit_idx < len(bits):
            raise ValueError(
                f"Frame has insufficient static capacity: "
                f"needed {len(bits)} bits, only {bit_idx} could be embedded"
            )

        ycrcb[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(
        self,
        stego_frame: np.ndarray,
        prev_frame: np.ndarray,
        n_bits: int,
    ) -> np.ndarray:
        """Extract *n_bits* bits from a stego frame.

        Parameters
        ----------
        stego_frame:
            BGR ``uint8`` stego frame.
        prev_frame:
            The same preceding frame used during embedding.
        n_bits:
            Number of bits to extract.

        Returns
        -------
        np.ndarray
            1-D ``uint8`` array of extracted bits.
        """
        motion_mask = self._analyzer.motion_mask(prev_frame, stego_frame)
        ycrcb = cv2.cvtColor(stego_frame, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float64)

        h, w = y.shape
        extracted: List[int] = []

        for r in range(0, h - self.block_size + 1, self.block_size):
            for c in range(0, w - self.block_size + 1, self.block_size):
                if len(extracted) >= n_bits:
                    break
                block_mask = motion_mask[r : r + self.block_size, c : c + self.block_size]
                if block_mask.any():
                    continue
                block = y[r : r + self.block_size, c : c + self.block_size]
                mean_val = block.mean()
                extracted.append(self._read_bit(mean_val))

        return np.array(extracted[:n_bits], dtype=np.uint8)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def embed_message(
        self, frame: np.ndarray, prev_frame: np.ndarray, message: bytes
    ) -> np.ndarray:
        """Embed a complete message (with header) into *frame*."""
        frag = MessageFragmenter()
        bitstream = frag.build_bitstream(message)
        return self.embed(frame, prev_frame, bitstream)

    def extract_message(
        self, stego_frame: np.ndarray, prev_frame: np.ndarray
    ) -> bytes:
        """Extract a message embedded by :meth:`embed_message`."""
        frag = MessageFragmenter()
        header_bits = self.extract(stego_frame, prev_frame, frag.HEADER_BITS)
        n_payload = int(
            sum(int(b) << (frag.HEADER_BITS - 1 - i) for i, b in enumerate(header_bits))
        )
        total_bits = frag.HEADER_BITS + n_payload
        all_bits = self.extract(stego_frame, prev_frame, total_bits)
        return frag.parse_bitstream(all_bits)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adjust_mean(self, mean_val: float, bit: int) -> float:
        """Round *mean_val* toward the nearest value whose parity equals *bit*."""
        rounded = round(mean_val)
        if rounded % 2 != bit:
            # Try both neighbours and pick the closer one
            candidate_a = rounded - 1
            candidate_b = rounded + 1
            if abs(candidate_a - mean_val) <= abs(candidate_b - mean_val):
                rounded = candidate_a
            else:
                rounded = candidate_b
        return float(rounded)

    def _read_bit(self, mean_val: float) -> int:
        """Decode the bit encoded in *mean_val* via its parity."""
        return int(round(mean_val)) % 2
