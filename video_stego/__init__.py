"""
video_stego — 视频帧间信息隐藏实验平台
Video inter-frame information hiding experiment platform.
"""

from .codec import VideoDecoder, VideoEncoder
from .sampling import FrameSampler
from .motion import MotionAnalyzer
from .message import MessageFragmenter
from .metrics import compute_psnr, compute_ssim, compute_ber
from .experiment import ExperimentManager

__all__ = [
    "VideoDecoder",
    "VideoEncoder",
    "FrameSampler",
    "MotionAnalyzer",
    "MessageFragmenter",
    "compute_psnr",
    "compute_ssim",
    "compute_ber",
    "ExperimentManager",
]
