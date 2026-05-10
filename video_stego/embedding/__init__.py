"""
video_stego.embedding — steganographic embedding methods.
"""

from .pixel import PixelEmbedder
from .dct import DCTEmbedder
from .interframe import InterFrameEmbedder

__all__ = ["PixelEmbedder", "DCTEmbedder", "InterFrameEmbedder"]
