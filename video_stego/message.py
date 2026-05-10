"""
message.py — 秘密消息分片模块
Secret message fragmentation and bit-level utilities.

:class:`MessageFragmenter` converts an arbitrary byte string into a flat bit
array and splits it into chunks that can be embedded across multiple frames or
regions.
"""

from __future__ import annotations

from typing import List

import numpy as np


class MessageFragmenter:
    """Convert and fragment a secret message for distributed embedding.

    The message is serialised as:

    * A 32-bit big-endian integer containing the total number of payload bits.
    * The payload bits themselves (MSB-first per byte).

    The length header lets the extractor know exactly how many bits to collect
    before stopping, even if the container capacity is larger.

    Examples
    --------
    >>> frag = MessageFragmenter()
    >>> chunks = frag.fragment(b"hello", chunk_size=20)
    >>> recovered = frag.assemble(chunks)
    >>> assert recovered == b"hello"
    """

    HEADER_BITS = 32  # bits used for the length header

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def bytes_to_bits(data: bytes) -> np.ndarray:
        """Convert *data* to a bit array (MSB-first, dtype ``uint8``)."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return bits.astype(np.uint8)

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        """Pack a bit array back to bytes (MSB-first).

        The length of *bits* is rounded up to a multiple of 8 by zero-padding.
        """
        bits = np.asarray(bits, dtype=np.uint8)
        pad = (-len(bits)) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        return np.packbits(bits).tobytes()

    # ------------------------------------------------------------------
    # Fragmentation / assembly
    # ------------------------------------------------------------------

    def build_bitstream(self, message: bytes) -> np.ndarray:
        """Encode *message* into a self-delimiting bit array.

        The layout is: ``[32-bit length header][payload bits]``.

        Returns
        -------
        np.ndarray
            1-D ``uint8`` array of ``32 + 8 * len(message)`` bits.
        """
        payload_bits = self.bytes_to_bits(message)
        n_bits = len(payload_bits)
        header = np.array(
            [(n_bits >> i) & 1 for i in reversed(range(self.HEADER_BITS))],
            dtype=np.uint8,
        )
        return np.concatenate([header, payload_bits])

    def parse_bitstream(self, bitstream: np.ndarray) -> bytes:
        """Decode a self-delimiting bit array produced by :meth:`build_bitstream`.

        Parameters
        ----------
        bitstream:
            A 1-D ``uint8`` array that starts with the 32-bit length header.

        Returns
        -------
        bytes
            The original message payload.

        Raises
        ------
        ValueError
            If the bitstream is shorter than the header + declared payload.
        """
        if len(bitstream) < self.HEADER_BITS:
            raise ValueError("Bitstream too short to contain header")
        n_bits = int(
            sum(int(b) << (self.HEADER_BITS - 1 - i) for i, b in enumerate(bitstream[: self.HEADER_BITS]))
        )
        end = self.HEADER_BITS + n_bits
        if len(bitstream) < end:
            raise ValueError(
                f"Bitstream declares {n_bits} payload bits but only "
                f"{len(bitstream) - self.HEADER_BITS} are available"
            )
        return self.bits_to_bytes(bitstream[self.HEADER_BITS : end])

    def fragment(self, message: bytes, chunk_size: int) -> List[np.ndarray]:
        """Split the encoded bitstream into chunks of at most *chunk_size* bits.

        Parameters
        ----------
        message:
            The secret payload to hide.
        chunk_size:
            Maximum number of bits per chunk.

        Returns
        -------
        list of np.ndarray
            The bitstream divided into chunks (the last chunk may be smaller).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        bitstream = self.build_bitstream(message)
        chunks = []
        for start in range(0, len(bitstream), chunk_size):
            chunks.append(bitstream[start : start + chunk_size])
        return chunks

    def assemble(self, chunks: List[np.ndarray]) -> bytes:
        """Reassemble fragments and decode the original message.

        Parameters
        ----------
        chunks:
            Ordered list of bit-array fragments as returned by
            :meth:`fragment` (or extracted during decoding).

        Returns
        -------
        bytes
            The recovered message payload.
        """
        bitstream = np.concatenate(chunks).astype(np.uint8)
        return self.parse_bitstream(bitstream)

    # ------------------------------------------------------------------
    # Capacity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def max_message_bytes(capacity_bits: int) -> int:
        """Maximum message length in bytes given a container capacity.

        Accounts for the 32-bit header overhead.

        Parameters
        ----------
        capacity_bits:
            Total number of embeddable bits.

        Returns
        -------
        int
            Maximum number of payload bytes that can be embedded.
        """
        payload_bits = max(0, capacity_bits - MessageFragmenter.HEADER_BITS)
        return payload_bits // 8
