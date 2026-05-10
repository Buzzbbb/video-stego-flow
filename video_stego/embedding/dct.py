"""
embedding/dct.py — 频域系数嵌入模块 (DCT-domain steganography)
Frequency-domain steganography: embed bits by quantised modification of DCT
coefficients in 8×8 blocks of the Y (luminance) channel.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..message import MessageFragmenter


class DCTEmbedder:
    """Embed and extract a secret message via DCT coefficient quantisation.

    The Y channel of each frame is divided into non-overlapping 8×8 blocks.
    Inside each block a designated mid-frequency coefficient (the ``coeff_pos``
    element in zig-zag order) is modified so that its value modulo
    ``2 * quantisation_step`` encodes the desired bit:

    * bit = 0 → coefficient made *even* multiple of ``quantisation_step``
    * bit = 1 → coefficient made *odd* multiple of ``quantisation_step``

    Parameters
    ----------
    quantisation_step:
        Granularity of quantisation.  Larger values give better robustness
        but introduce more distortion.
    coeff_index:
        Index of the DCT coefficient (in raster order within the 8×8 block)
        used for embedding.  Default is 5, which selects a low-to-mid
        frequency coefficient.

    Examples
    --------
    >>> emb = DCTEmbedder(quantisation_step=20)
    >>> stego = emb.embed(frame, bits)
    >>> recovered = emb.extract(stego, len(bits))
    """

    BLOCK = 8  # Block side length (pixels)

    def __init__(
        self,
        quantisation_step: int = 20,
        coeff_index: int = 5,
    ) -> None:
        if quantisation_step <= 0:
            raise ValueError("quantisation_step must be positive")
        if not 0 <= coeff_index < self.BLOCK * self.BLOCK:
            raise ValueError(
                f"coeff_index must be in [0, {self.BLOCK ** 2 - 1}]"
            )
        self.quantisation_step = quantisation_step
        self.coeff_index = coeff_index
        # Raster position of the target coefficient
        self._cr = coeff_index // self.BLOCK
        self._cc = coeff_index % self.BLOCK

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    def capacity_bits(self, frame: np.ndarray) -> int:
        """Return the number of embeddable bits in *frame* (one per block)."""
        h, w = frame.shape[:2]
        n_blocks_h = h // self.BLOCK
        n_blocks_w = w // self.BLOCK
        return n_blocks_h * n_blocks_w

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed(self, frame: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """Embed *bits* into the DCT coefficients of *frame*.

        Parameters
        ----------
        frame:
            Cover BGR ``uint8`` array.
        bits:
            1-D ``uint8`` bit array to embed (values 0 or 1).

        Returns
        -------
        np.ndarray
            Stego BGR ``uint8`` frame.
        """
        cap = self.capacity_bits(frame)
        if len(bits) > cap:
            raise ValueError(
                f"Cannot embed {len(bits)} bits; frame capacity is {cap} bits"
            )

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_float = ycrcb[:, :, 0].astype(np.float32)

        h, w = y_float.shape
        bit_idx = 0
        bits = np.asarray(bits, dtype=np.uint8)

        for row in range(0, h - self.BLOCK + 1, self.BLOCK):
            for col in range(0, w - self.BLOCK + 1, self.BLOCK):
                if bit_idx >= len(bits):
                    break
                block = y_float[row : row + self.BLOCK, col : col + self.BLOCK]
                dct_block = cv2.dct(block)
                dct_block = self._embed_bit(dct_block, int(bits[bit_idx]))
                y_float[row : row + self.BLOCK, col : col + self.BLOCK] = cv2.idct(dct_block)
                bit_idx += 1

        y_float = np.clip(y_float, 0, 255)
        ycrcb[:, :, 0] = y_float.astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(self, stego_frame: np.ndarray, n_bits: int) -> np.ndarray:
        """Extract *n_bits* bits from a DCT-stego frame.

        Parameters
        ----------
        stego_frame:
            BGR ``uint8`` stego frame.
        n_bits:
            Number of bits to extract.

        Returns
        -------
        np.ndarray
            1-D ``uint8`` array of extracted bits (values 0 or 1).
        """
        ycrcb = cv2.cvtColor(stego_frame, cv2.COLOR_BGR2YCrCb)
        y_float = ycrcb[:, :, 0].astype(np.float32)

        h, w = y_float.shape
        extracted: List[int] = []

        for row in range(0, h - self.BLOCK + 1, self.BLOCK):
            for col in range(0, w - self.BLOCK + 1, self.BLOCK):
                if len(extracted) >= n_bits:
                    break
                block = y_float[row : row + self.BLOCK, col : col + self.BLOCK]
                dct_block = cv2.dct(block)
                extracted.append(self._extract_bit(dct_block))

        return np.array(extracted[:n_bits], dtype=np.uint8)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def embed_message(self, frame: np.ndarray, message: bytes) -> np.ndarray:
        """Embed a complete message (with header) into *frame*."""
        frag = MessageFragmenter()
        bitstream = frag.build_bitstream(message)
        return self.embed(frame, bitstream)

    def extract_message(self, stego_frame: np.ndarray) -> bytes:
        """Extract a message embedded by :meth:`embed_message`."""
        frag = MessageFragmenter()
        header_bits = self.extract(stego_frame, frag.HEADER_BITS)
        n_payload = int(
            sum(int(b) << (frag.HEADER_BITS - 1 - i) for i, b in enumerate(header_bits))
        )
        total_bits = frag.HEADER_BITS + n_payload
        all_bits = self.extract(stego_frame, total_bits)
        return frag.parse_bitstream(all_bits)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_bit(self, dct_block: np.ndarray, bit: int) -> np.ndarray:
        """Modify dct_block[_cr, _cc] to encode *bit* via quantisation."""
        q = self.quantisation_step
        coeff = dct_block[self._cr, self._cc]
        coeff_q = round(coeff / q)
        # Make coeff_q even (bit=0) or odd (bit=1)
        if bit == 0:
            if coeff_q % 2 != 0:
                coeff_q += 1
        else:
            if coeff_q % 2 == 0:
                coeff_q += 1
        dct_block = dct_block.copy()
        dct_block[self._cr, self._cc] = float(coeff_q * q)
        return dct_block

    def _extract_bit(self, dct_block: np.ndarray) -> int:
        """Read the embedded bit from dct_block[_cr, _cc]."""
        q = self.quantisation_step
        coeff = dct_block[self._cr, self._cc]
        coeff_q = round(coeff / q)
        return int(abs(coeff_q) % 2)
