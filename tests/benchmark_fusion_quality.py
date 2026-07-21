"""
Fusion quality benchmark - terminal tables for every fusion method.

Three modes:

* sweep (default) - one method across its tuning parameter
* --all           - every available method at its default setting
* --list          - what is registered and whether it can run here

With a real stack only the no-reference metrics apply (Q_ABF is the one to
watch). With --synthetic an all-in-focus ground truth exists, so PSNR and SSIM
are reported too.

Examples:
    python tests/benchmark_fusion_quality.py --synthetic
    python tests/benchmark_fusion_quality.py --synthetic --all
    python tests/benchmark_fusion_quality.py --synthetic --method dct
    python tests/benchmark_fusion_quality.py path/to/stack --method dtcwt --values 2,3,4,5
    python tests/benchmark_fusion_quality.py path/to/stack --all --save-dir out
"""

import argparse
import glob
import os
import re
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fusion_metrics as fm
from tests import fusion_registry as reg
from tests.synthetic_stack import make_stack

# Metric key -> (column header, format, higher-is-better)
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
    paths = [p for p in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isfile(p)]
    try:
        paths.sort(key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1]))
    except IndexError:
        paths.sort()

    stack = [img for img in (cv2.imread(p) for p in paths) if img is not None]
    if not stack:
        raise SystemExit(f"No readable images found in {path}")
    return stack


def plan_runs(method_key, run_all, values_override):
    """Return (runs, axis_label) where each run is (label, slug, fuse_callable)."""
    if run_all:
        methods = reg.available_methods()
        if not methods:
            raise SystemExit("No fusion method is available in this environment")
        return [(m.label, m.key, m.run) for m in methods], "Method"

    method = reg.get(method_key)
    ok, why = method.available()
    if not ok:
        raise SystemExit(f"{method.label} is unavailable: {why}")

    if method.sweep is None:
        return [(method.label, method.key, method.run)], "Method"

    param, values = method.sweep
    if values_override:
        values = values_override

    def make(value):
        return lambda stack, **kw: method.run(stack, **{param: value}, **kw)

    # Keeping the parameter in the slug means saved files say what produced them
    return ([(str(v), f"{method.key}_{param}{v}", make(v)) for v in values], param)


def print_table(rows, axis_label, has_reference):
    """Print the results grid, marking the best value in each metric column."""
    cols = [c for c in COLUMNS if has_reference or c[0] not in ("psnr", "ssim")]
    width = max(12, max(len(r["label"]) for r in rows) + 2)

    header = f"{axis_label:<{width}}" + "".join(f"{c[1]:>11}" for c in cols)
    print(header)
    print("-" * len(header))

    # Find the winning run per metric so the table points at an answer
    best = {}
    for key, _, _, higher_better in cols:
        if not higher_better:
            continue
        candidates = [(r[key], r["label"]) for r in rows if key in r]
        if candidates:
            best[key] = max(candidates)[1]

    for r in rows:
        line = f"{r['label']:<{width}}"
        for key, _, fmt, _ in cols:
            if key not in r:
                line += f"{'-':>11}"
                continue
            cell = fmt.format(r[key])
            if best.get(key) == r["label"]:
                cell += "*"
            line += f"{cell:>11}"
        print(line)

    if best:
        print("\n* = best in that column")
        print(f"Best Q_ABF: {best.get('qabf')}")
        if has_reference:
            print(f"Best PSNR:  {best.get('psnr')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", nargs="?",
                    help="Directory of focus-stack images (omit with --synthetic)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use a generated stack with a known all-in-focus reference")
    ap.add_argument("--method", default="guided_filter",
                    help="Fusion method key; see --list")
    ap.add_argument("--all", action="store_true", dest="run_all",
                    help="Benchmark every available method instead of sweeping one")
    ap.add_argument("--list", action="store_true", dest="list_methods",
                    help="List registered methods and their availability, then exit")
    ap.add_argument("--slices", type=int, default=5, help="Slices for --synthetic")
    ap.add_argument("--size", type=int, default=512, help="Edge length for --synthetic")
    ap.add_argument("--style", default="photographic", choices=("photographic", "texture"),
                    help="Synthetic reference style (default: photographic)")
    ap.add_argument("--values", default=None,
                    help="Comma-separated override for the swept parameter")
    ap.add_argument("--save-dir", help="Write each fused result here as <slug>.png")
    args = ap.parse_args()

    if args.list_methods:
        print(f"{'key':<20}{'status':<9}{'sweep':<28}reason")
        for m in reg.METHODS:
            ok, why = m.available()
            sweep = f"{m.sweep[0]}={m.sweep[1]}" if m.sweep else "-"
            print(f"{m.key:<20}{'ready' if ok else 'skip':<9}{sweep:<28}{'' if ok else why}")
        return

    if not args.synthetic and not args.stack:
        ap.error("provide a stack directory or pass --synthetic")

    values = [int(v) for v in args.values.split(",") if v.strip()] if args.values else None

    if args.synthetic:
        stack, reference, _ = make_stack(num_slices=args.slices, height=args.size,
                                         width=args.size, seed=7, style=args.style)
        source = f"synthetic {args.style} ({args.slices} slices, {args.size}x{args.size})"
    else:
        stack = load_stack(args.stack)
        reference = None
        source = (f"{args.stack} ({len(stack)} slices, "
                  f"{stack[0].shape[1]}x{stack[0].shape[0]})")

    runs, axis_label = plan_runs(args.method, args.run_all, values)
    print(f"Fusion quality benchmark\nStack: {source}\n")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    rows = []
    for label, slug, fuse in runs:
        start = time.perf_counter()
        try:
            fused = fuse(stack)
        except Exception as exc:  # a failing method must not sink the run
            print(f"  {label:<30} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        elapsed = time.perf_counter() - start

        row = {"label": label, "seconds": elapsed}
        # DCT and DTCWT can return a slightly different geometry than they were
        # given, so measure on the region every image actually shares
        m_fused, m_stack, m_ref = fm.align_to_common_size(fused, stack, reference)
        row.update(fm.evaluate(m_fused, m_stack, m_ref))
        rows.append(row)

        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, f"{slug}.png"), fused)

    if not rows:
        raise SystemExit("Every fusion run failed; nothing to report")

    print()
    print_table(rows, axis_label, has_reference=reference is not None)

    if args.save_dir:
        print(f"\nFused images written to {args.save_dir}")


if __name__ == "__main__":
    main()
