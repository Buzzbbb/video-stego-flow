"""
embedding/pixel.py — 帧内像素域嵌入模块 (LSB steganography)
Pixel-domain steganography: embed bits in the least-significant bits of
luminance (Y channel) pixel values.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..message import MessageFragmenter


class PixelEmbedder:
    """Embed and extract a secret message using LSB pixel substitution.

    The message is written into the least-significant *lsb_count* bits of the
    Y (luminance) channel of each frame, scanning raster-order from the
    top-left corner.

    Parameters
    ----------
    lsb_count:
        Number of least-significant bits per pixel to use for embedding.
        Must be in the range [1, 4].

    Examples
    --------
    >>> emb = PixelEmbedder(lsb_count=1)
    >>> stego_frame = emb.embed(cover_frame, message_bits)
    >>> recovered_bits = emb.extract(stego_frame, n_bits)
    """

    def __init__(self, lsb_count: int = 1) -> None:
        if not 1 <= lsb_count <= 4:
            raise ValueError("lsb_count must be between 1 and 4")
        self.lsb_count = lsb_count
        self._mask = np.uint8((1 << lsb_count) - 1)
        self._clear_mask = np.uint8(0xFF ^ self._mask)

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    def capacity_bits(self, frame: np.ndarray) -> int:
        """Return the number of bits embeddable in *frame*."""
        h, w = frame.shape[:2]
        return h * w * self.lsb_count

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed(self, frame: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """Return a stego frame with *bits* embedded in the LSBs.

        Parameters
        ----------
        frame:
            Cover BGR ``uint8`` array.
        bits:
            1-D ``uint8`` array of bits to embed (values 0 or 1).

        Returns
        -------
        np.ndarray
            Stego frame (BGR ``uint8``) with the same shape as *frame*.
        """
        cap = self.capacity_bits(frame)
        if len(bits) > cap:
            raise ValueError(
                f"Cannot embed {len(bits)} bits; frame capacity is {cap} bits"
            )

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].copy().astype(np.uint8)
        flat_y = y.flatten()

        # Build bit groups of size lsb_count
        bits = np.asarray(bits, dtype=np.uint8)
        # Pad bits to a multiple of lsb_count
        pad = (-len(bits)) % self.lsb_count
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

        # Each pixel holds lsb_count bits
        n_pixels = len(bits) // self.lsb_count
        groups = bits.reshape(n_pixels, self.lsb_count)
        # Convert each group to an integer value
        powers = 1 << np.arange(self.lsb_count - 1, -1, -1, dtype=np.uint8)
        values = (groups * powers).sum(axis=1).astype(np.uint8)

        # Clear LSBs and embed
        flat_y[:n_pixels] = (flat_y[:n_pixels] & self._clear_mask) | values

        ycrcb[:, :, 0] = flat_y.reshape(y.shape)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(self, stego_frame: np.ndarray, n_bits: int) -> np.ndarray:
        """Extract *n_bits* bits from *stego_frame*.

        Parameters
        ----------
        stego_frame:
            BGR ``uint8`` stego frame.
        n_bits:
            Number of bits to extract (must match the value used during
            embedding).

        Returns
        -------
        np.ndarray
            1-D ``uint8`` array of length *n_bits* (values 0 or 1).
        """
        ycrcb = cv2.cvtColor(stego_frame, cv2.COLOR_BGR2YCrCb)
        flat_y = ycrcb[:, :, 0].flatten().astype(np.uint8)

        # How many pixels do we need?
        n_pixels = (n_bits + self.lsb_count - 1) // self.lsb_count
        if n_pixels > len(flat_y):
            raise ValueError("stego_frame too small to hold the declared bits")

        pixel_vals = flat_y[:n_pixels] & self._mask
        # Unpack each pixel value into lsb_count bits
        powers = 1 << np.arange(self.lsb_count - 1, -1, -1, dtype=np.uint8)
        bits = ((pixel_vals[:, None] & powers) > 0).astype(np.uint8).flatten()
        return bits[:n_bits]

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def embed_message(self, frame: np.ndarray, message: bytes) -> np.ndarray:
        """Embed a complete message (with header) into *frame*.

        Uses :class:`~video_stego.message.MessageFragmenter` to encode the
        message into a self-delimiting bitstream before embedding.
        """
        frag = MessageFragmenter()
        bitstream = frag.build_bitstream(message)
        return self.embed(frame, bitstream)

    def extract_message(self, stego_frame: np.ndarray) -> bytes:
        """Extract and decode a message embedded by :meth:`embed_message`."""
        frag = MessageFragmenter()
        header_bits = self.extract(stego_frame, frag.HEADER_BITS)
        n_payload = int(
            sum(int(b) << (frag.HEADER_BITS - 1 - i) for i, b in enumerate(header_bits))
        )
        total_bits = frag.HEADER_BITS + n_payload
        all_bits = self.extract(stego_frame, total_bits)
        return frag.parse_bitstream(all_bits)
