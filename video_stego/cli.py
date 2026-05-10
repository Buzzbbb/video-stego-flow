"""
cli.py — 命令行入口
Command-line interface for the video steganography experiment platform.

Usage examples
--------------
Embed a message with LSB pixel steganography::

    video-stego embed --method pixel --lsb-count 1 \\
        --input cover.mp4 --output stego.mp4 \\
        --message "secret text"

Extract the message::

    video-stego extract --method pixel --lsb-count 1 \\
        --input stego.mp4

Run a robustness test (embed then attack and measure BER / PSNR)::

    video-stego robustness --method pixel --attack compress \\
        --quality 70 --input cover.mp4 --message "secret text"

List saved experiments::

    video-stego experiments list --storage experiments.json
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-stego",
        description="视频帧间信息隐藏实验平台 — Video Steganography Experiment Platform",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- embed ----
    p_embed = sub.add_parser("embed", help="Embed a message into a video")
    _add_io_args(p_embed)
    _add_embed_args(p_embed)
    p_embed.add_argument("--message", required=True, help="Secret message text")
    p_embed.add_argument("--frame-limit", type=int, default=None,
                         help="Maximum number of frames to process")

    # ---- extract ----
    p_extract = sub.add_parser("extract", help="Extract a message from a stego video")
    p_extract.add_argument("--input", required=True, metavar="VIDEO")
    _add_embed_args(p_extract)

    # ---- robustness ----
    p_robust = sub.add_parser(
        "robustness", help="Embed, attack, then measure BER and PSNR"
    )
    _add_io_args(p_robust)
    _add_embed_args(p_robust)
    p_robust.add_argument("--message", required=True, help="Secret message text")
    p_robust.add_argument(
        "--attack",
        choices=["compress", "crop", "framedrop", "transcode", "none"],
        default="compress",
    )
    p_robust.add_argument("--quality", type=int, default=75,
                          help="JPEG quality for compression attack (1-100)")
    p_robust.add_argument("--drop-rate", type=float, default=0.1,
                          help="Fraction of frames to drop")
    p_robust.add_argument("--crop-pixels", type=int, default=10,
                          help="Pixels to crop from each edge")
    p_robust.add_argument("--save", metavar="JSON",
                          help="Save experiment results to this JSON file")
    p_robust.add_argument("--frame-limit", type=int, default=None)

    # ---- experiments ----
    p_exp = sub.add_parser("experiments", help="List saved experiments")
    p_exp.add_argument("action", choices=["list"], help="Action to perform")
    p_exp.add_argument("--storage", default="experiments.json",
                       help="Path to the experiments JSON file")

    return parser


def _add_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True, metavar="VIDEO", help="Input video path")
    p.add_argument("--output", metavar="VIDEO", help="Output stego video path")


def _add_embed_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--method",
        choices=["pixel", "dct", "interframe"],
        default="pixel",
        help="Steganography method",
    )
    p.add_argument("--lsb-count", type=int, default=1,
                   help="Number of LSBs per pixel (pixel method)")
    p.add_argument("--quantisation-step", type=int, default=20,
                   help="DCT quantisation step (dct method)")
    p.add_argument("--delta", type=int, default=3,
                   help="Embedding perturbation delta (interframe method)")
    p.add_argument("--block-size", type=int, default=8,
                   help="Block size for DCT / interframe methods")


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _load_frames(path: str, limit: int | None = None) -> tuple:
    """Return (frames, fps, (w, h))."""
    from .codec import VideoDecoder
    dec = VideoDecoder(path)
    frames = dec.read_frames(end=limit)
    return frames, dec.fps, dec.frame_size


def _make_embedder(args: argparse.Namespace):
    """Instantiate the requested embedder from CLI args."""
    from .embedding import PixelEmbedder, DCTEmbedder, InterFrameEmbedder
    if args.method == "pixel":
        return PixelEmbedder(lsb_count=args.lsb_count)
    if args.method == "dct":
        return DCTEmbedder(
            quantisation_step=args.quantisation_step,
            coeff_index=5,
        )
    return InterFrameEmbedder(
        delta=args.delta,
        block_size=args.block_size,
    )


def _cmd_embed(args: argparse.Namespace) -> None:
    from .codec import VideoEncoder
    frames, fps, frame_size = _load_frames(args.input, args.frame_limit)
    if not frames:
        print("ERROR: no frames decoded from input video", file=sys.stderr)
        sys.exit(1)

    embedder = _make_embedder(args)
    message = args.message.encode("utf-8")
    stego_frames = list(frames)

    if args.method == "interframe":
        for i in range(1, len(frames)):
            stego_frames[i] = embedder.embed_message(frames[i], frames[i - 1], message)
            break  # embed into first eligible frame only for simplicity
    else:
        stego_frames[0] = embedder.embed_message(frames[0], message)

    output = args.output or "stego_output.mp4"
    enc = VideoEncoder(output, fps=fps, frame_size=frame_size)
    enc.write_frames(stego_frames)
    print(f"Stego video written to: {output}")


def _cmd_extract(args: argparse.Namespace) -> None:
    frames, _fps, _size = _load_frames(args.input)
    if not frames:
        print("ERROR: no frames decoded from input video", file=sys.stderr)
        sys.exit(1)

    embedder = _make_embedder(args)
    if args.method == "interframe":
        if len(frames) < 2:
            print("ERROR: need at least 2 frames for interframe extraction", file=sys.stderr)
            sys.exit(1)
        message = embedder.extract_message(frames[1], frames[0])
    else:
        message = embedder.extract_message(frames[0])
    print(f"Extracted message: {message.decode('utf-8', errors='replace')}")


def _cmd_robustness(args: argparse.Namespace) -> None:
    from .message import MessageFragmenter
    from .metrics import compute_psnr, compute_ssim, compute_ber
    from .experiment import ExperimentManager

    frames, fps, frame_size = _load_frames(args.input, args.frame_limit)
    if not frames:
        print("ERROR: no frames decoded from input video", file=sys.stderr)
        sys.exit(1)

    embedder = _make_embedder(args)
    message = args.message.encode("utf-8")
    frag = MessageFragmenter()
    original_bits = frag.build_bitstream(message)

    # Embed
    stego_frames = list(frames)
    if args.method == "interframe":
        stego_frames[1] = embedder.embed_message(frames[1], frames[0], message)
    else:
        stego_frames[0] = embedder.embed_message(frames[0], message)

    # Apply attack
    attacked_frames = _apply_attack(args, stego_frames)

    # Extract after attack
    if args.method == "interframe":
        if len(attacked_frames) < 2:
            print("ERROR: too few frames after attack", file=sys.stderr)
            sys.exit(1)
        try:
            recovered = embedder.extract_message(attacked_frames[1], attacked_frames[0])
        except Exception as exc:
            print(f"Extraction failed: {exc}", file=sys.stderr)
            recovered = b""
    else:
        try:
            recovered = embedder.extract_message(attacked_frames[0])
        except Exception as exc:
            print(f"Extraction failed: {exc}", file=sys.stderr)
            recovered = b""

    recovered_bits = frag.build_bitstream(recovered) if recovered else np.zeros_like(original_bits)

    # Compute metrics
    min_len = min(len(original_bits), len(recovered_bits))
    ber = compute_ber(original_bits[:min_len], recovered_bits[:min_len])
    psnr = compute_psnr(frames[0], attacked_frames[0])
    ssim = compute_ssim(frames[0], attacked_frames[0])

    print(f"Attack         : {args.attack}")
    print(f"Method         : {args.method}")
    print(f"PSNR           : {psnr:.2f} dB")
    print(f"SSIM           : {ssim:.4f}")
    print(f"BER            : {ber:.4f} ({ber * 100:.2f}%)")
    print(f"Recovered msg  : {recovered.decode('utf-8', errors='replace')!r}")

    if args.save:
        mgr = ExperimentManager(args.save)
        exp_id = mgr.save(
            params={
                "method": args.method,
                "attack": args.attack,
                "lsb_count": getattr(args, "lsb_count", None),
                "quantisation_step": getattr(args, "quantisation_step", None),
                "quality": getattr(args, "quality", None),
            },
            metrics={"psnr": psnr, "ssim": ssim, "ber": ber},
            notes=f"robustness test: {args.method} + {args.attack}",
        )
        print(f"Experiment saved with ID: {exp_id}")


def _apply_attack(args: argparse.Namespace, frames: list) -> list:
    if args.attack == "none":
        return frames
    if args.attack == "compress":
        from .attack import CompressionAttack
        return CompressionAttack(quality=args.quality).apply(frames)
    if args.attack == "crop":
        from .attack import CropAttack
        return CropAttack(crop_pixels=args.crop_pixels, resize_back=True).apply(frames)
    if args.attack == "framedrop":
        from .attack import FrameDropAttack
        return FrameDropAttack(drop_rate=args.drop_rate, fill_gaps=True, seed=0).apply(frames)
    if args.attack == "transcode":
        from .attack import TranscodeAttack
        return TranscodeAttack().apply(frames)
    return frames


def _cmd_experiments(args: argparse.Namespace) -> None:
    from .experiment import ExperimentManager
    mgr = ExprimentManager = ExperimentManager(args.storage)
    print(mgr.summary())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    if args.command == "embed":
        _cmd_embed(args)
    elif args.command == "extract":
        _cmd_extract(args)
    elif args.command == "robustness":
        _cmd_robustness(args)
    elif args.command == "experiments":
        _cmd_experiments(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
