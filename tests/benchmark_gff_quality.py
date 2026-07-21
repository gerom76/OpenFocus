"""
Guided Filter fusion quality benchmark.

Fuses a focus stack with fusion_methods/gff.py across a range of kernel sizes
and reports the fusion metrics plus runtime for each, so the kernel slider can
be tuned against numbers instead of eyeballing.

With a real stack only the no-reference metrics apply (Q^AB/F is the one to
watch). With --synthetic an all-in-focus ground truth exists, so PSNR and SSIM
are reported too.

Examples:
    python tests/benchmark_gff_quality.py --synthetic
    python tests/benchmark_gff_quality.py path/to/stack --kernels 7,15,31,63
    python tests/benchmark_gff_quality.py path/to/stack --save-dir out --compare-torch
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_methods.gff import gff_impl
from tests import fusion_metrics as fm
from tests.synthetic_stack import make_stack

DEFAULT_KERNELS = [7, 15, 31, 45, 63]

# Metric name -> (column header, format, higher-is-better)
COLUMNS = [
    ("qabf", "Q_ABF", "{:.4f}", True),
    ("entropy", "Entropy", "{:.3f}", True),
    ("spatial_frequency", "SpatFreq", "{:.3f}", True),
    ("std_dev", "StdDev", "{:.2f}", True),
    ("psnr", "PSNR", "{:.2f}", True),
    ("ssim", "SSIM", "{:.4f}", True),
    ("seconds", "Time(s)", "{:.2f}", False),
]


def load_stack(path):
    """Load every readable image in a directory, sorted numerically by filename."""
    import glob
    import re

    paths = sorted(glob.glob(os.path.join(path, "*")))
    paths = [p for p in paths if os.path.isfile(p)]
    try:
        paths.sort(key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1]))
    except IndexError:
        paths.sort()

    stack = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            stack.append(img)
    if not stack:
        raise SystemExit(f"No readable images in {path}")
    return stack


def print_table(rows, has_reference):
    """Print the results grid, marking the best value in each metric column."""
    cols = [c for c in COLUMNS
            if has_reference or c[0] not in ("psnr", "ssim")]

    header = f"{'Kernel':>7}" + "".join(f"{c[1]:>11}" for c in cols)
    print(header)
    print("-" * len(header))

    # Find the winning kernel per metric so the table points at an answer
    best = {}
    for key, _, _, higher_better in cols:
        if not higher_better:
            continue
        vals = [(r[key], r["kernel"]) for r in rows if key in r]
        if vals:
            best[key] = max(vals)[1]

    for r in rows:
        line = f"{r['kernel']:>7}"
        for key, _, fmt, _ in cols:
            if key not in r:
                line += f"{'-':>11}"
                continue
            cell = fmt.format(r[key])
            if best.get(key) == r["kernel"]:
                cell += "*"
            line += f"{cell:>11}"
        print(line)

    if best:
        print("\n* = best for that metric")
        print(f"Recommended kernel (Q_ABF): {best.get('qabf')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", nargs="?",
                    help="Directory of focus-stack images (omit with --synthetic)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use a generated stack with a known all-in-focus reference")
    ap.add_argument("--slices", type=int, default=5,
                    help="Slice count for --synthetic (default: 5)")
    ap.add_argument("--size", type=int, default=512,
                    help="Edge length for --synthetic (default: 512)")
    ap.add_argument("--kernels", default=",".join(map(str, DEFAULT_KERNELS)),
                    help="Comma-separated kernel sizes to sweep")
    ap.add_argument("--threads", type=int, default=None,
                    help="Thread count passed to gff_impl")
    ap.add_argument("--save-dir",
                    help="Write each fused result here as gff_k<kernel>.png")
    ap.add_argument("--compare-torch", action="store_true",
                    help="Also run fusion_methods/gff_torch.py and report GPU/CPU agreement")
    args = ap.parse_args()

    if not args.synthetic and not args.stack:
        ap.error("provide a stack directory or pass --synthetic")

    kernels = [int(k) for k in args.kernels.split(",") if k.strip()]

    if args.synthetic:
        stack, reference, _ = make_stack(num_slices=args.slices,
                                         height=args.size, width=args.size)
        source = f"synthetic ({args.slices} slices, {args.size}x{args.size})"
    else:
        stack = load_stack(args.stack)
        reference = None
        source = f"{args.stack} ({len(stack)} slices, "\
                 f"{stack[0].shape[1]}x{stack[0].shape[0]})"

    print(f"Guided Filter quality benchmark\nStack: {source}\n")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    rows = []
    for kernel in kernels:
        start = time.perf_counter()
        fused = gff_impl(stack, img_resize=None, kernel_size=kernel,
                         thread_count=args.threads)
        elapsed = time.perf_counter() - start

        row = {"kernel": kernel, "seconds": elapsed}
        row.update(fm.evaluate(fused, stack, reference))
        rows.append(row)

        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, f"gff_k{kernel}.png"), fused)

        if args.compare_torch:
            row["torch_psnr"] = _torch_agreement(stack, kernel, fused)

    print_table(rows, has_reference=reference is not None)

    if args.compare_torch:
        print("\nCPU vs GPU agreement (PSNR, dB - higher means closer):")
        for r in rows:
            val = r.get("torch_psnr")
            shown = "n/a" if val is None else f"{val:.2f}"
            print(f"  kernel {r['kernel']:>3}: {shown}")

    if args.save_dir:
        print(f"\nFused images written to {args.save_dir}")


def _torch_agreement(stack, kernel, cpu_fused):
    """PSNR between the GPU and CPU results, or None if torch is unavailable."""
    try:
        import torch
        from fusion_methods.gff_torch import gff_torch_impl
    except ImportError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_fused = gff_torch_impl(stack, img_resize=None, kernel_size=kernel,
                               device=device)
    return fm.psnr(gpu_fused, cpu_fused)


if __name__ == "__main__":
    main()
