# Fusion quality reports

Self-contained HTML pages measuring the fusion methods in `fusion_methods/`.
Every image is embedded as a data URI, so a report opens offline in any browser
with no server and no assets folder beside it.

Everything here except this file is **generated** - regenerate rather than edit.
Both tools write into this folder by default, creating it if needed.

Run from the project root:

```bash
cd f:\src\OpenFocus
```

Requires only what OpenFocus already needs (`numpy`, `opencv-python`). Methods
whose optional dependency or weights file is missing are skipped with a printed
reason instead of failing.

---

## What is in here

| File | What it answers |
|------|-----------------|
| `fusion_method_guide.html` | Which method should I use, and what is each one bad at? Written for a non-specialist. |
| `fusion_comparison_report.html` | How do all methods compare on one stack, at their default settings? |
| `pyramid_quality_report.html` | What was wrong with the pyramid's published choose-max rule, and does the 1.19.0 rework fix it? Before and after on every fixture, with the faulty pixels drawn rather than asserted. |
| `guided_filter_kernel_size_report.html` | What does the Guided Filter kernel slider actually change? |
| `gfgfgf_kernel_size_report.html` | Same question for GFG-FGF. |
| `dct_block_size_report.html` | What does the DCT block size change? |
| `dct_kernel_size_report.html` | What does the DCT median-filter kernel change? |
| `dtcwt_N_report.html` | What do extra DTCWT decomposition levels buy? |
| `pyramid_selectivity_report.html` | What does the pyramid's Selectivity control change, from averaging to the published choose-max? |
| `pyramid_energy_window_report.html` | What does the kernel slider change for the pyramid? |
| `gff_ifcnn_kernel_size_report.html` | The kernel of the guided filter that IFCNN then refines. |

The pyramid's other five controls are swept inside `pyramid_quality_report.html`
rather than given a page each; `--all-params --method pyramid` writes those pages
if you want them separately.

---

## Regenerate everything

```bash
# 1. The plain-language guide: every method over six characterisation scenarios
python tests/visualize_fusion_characteristics.py

# 2. All methods side by side at their defaults
python tests/visualize_fusion_quality.py --synthetic --size 320 --compare

# 3. One report per tunable parameter, per method
for m in guided_filter gfgfgf dct dtcwt gff_ifcnn; do
  python tests/visualize_fusion_quality.py --synthetic --size 320 --all-params --method $m
done

# 4. The pyramid: its two headline dials, then the before/after evidence page
python tests/visualize_fusion_quality.py --synthetic --size 320 --method pyramid --param selectivity
python tests/visualize_fusion_quality.py --synthetic --size 320 --method pyramid --param energy_window
python tests/visualize_pyramid_quality.py
```

On Windows `cmd`, step 3 is:

```bat
for %m in (guided_filter gfgfgf dct dtcwt gff_ifcnn) do python tests/visualize_fusion_quality.py --synthetic --size 320 --all-params --method %m
```

Those four steps reproduce this folder exactly, from empty.

---

## Individual reports

```bash
# The method guide, opened when it finishes; --include-gpu also profiles the GPU variants
python tests/visualize_fusion_characteristics.py --open
python tests/visualize_fusion_characteristics.py --include-gpu

# One parameter of one method
python tests/visualize_fusion_quality.py --synthetic --method guided_filter --param kernel_size --open
python tests/visualize_fusion_quality.py --synthetic --method gfgfgf        --param kernel_size --open
python tests/visualize_fusion_quality.py --synthetic --method dct           --param block_size  --open
python tests/visualize_fusion_quality.py --synthetic --method dct           --param kernel_size --open
python tests/visualize_fusion_quality.py --synthetic --method dtcwt         --param N           --open
python tests/visualize_fusion_quality.py --synthetic --method gff_ifcnn     --param kernel_size --open
python tests/visualize_fusion_quality.py --synthetic --method pyramid       --param selectivity --open

# Every parameter of one method, in one go
python tests/visualize_fusion_quality.py --synthetic --all-params --method dct

# The pyramid's before/after evidence page; --quick skips the sweeps, which are
# most of its 20 s
python tests/visualize_pyramid_quality.py --open
python tests/visualize_pyramid_quality.py --quick --open
```

`stackmffv4` exposes no tuning parameter, so it appears only in the comparison
report and the guide.

---

## Options worth knowing

| Option | Effect |
|--------|--------|
| `--list` | Every method, whether it can run here, and its tunable parameters. |
| `--param NAME` | Which parameter to sweep. Defaults to the method's first. |
| `--all-params` | One report per tunable parameter of `--method`. |
| `--values 3,4,5` | Override the swept values. |
| `--compare` | All methods at their defaults, instead of sweeping one. |
| `--synthetic` | Generate a stack with a known perfect answer, enabling PSNR and SSIM. |
| `--size N`, `--slices N` | Dimensions and frame count of the synthetic stack. |
| `--style photographic\|texture` | Synthetic scene type. See the caveat below. |
| `--out-dir DIR` | Where reports land. Defaults to `reports/`. |
| `--out PATH` | An exact output path, honoured as given. |
| `--open` | Open the report in the browser when it finishes. |

```bash
python tests/visualize_fusion_quality.py --list
```

---

## Your own images

Point the tool at a directory of stack frames instead of `--synthetic`:

```bash
python tests/visualize_fusion_quality.py path/to/stack --method dtcwt --param N --open
python tests/visualize_fusion_quality.py path/to/stack --compare --open
```

A real stack has no known perfect answer, so PSNR, SSIM and the error maps are
omitted and the report falls back to Q^AB/F - which measures how much source
edge detail survived and needs no ground truth. **This is the mode to trust when
choosing a setting for your own work**; the synthetic scenes are for regression
testing and for comparing methods on equal terms.

---

## Terminal tables

The same measurements without generating a page, useful for a quick check:

```bash
python tests/benchmark_fusion_quality.py --synthetic --all
python tests/benchmark_fusion_quality.py --synthetic --method dct --param block_size
python tests/benchmark_fusion_quality.py path/to/stack --method dtcwt --param N
python tests/benchmark_fusion_quality.py --synthetic --all --save-dir reports/frames
```

---

## Reading the numbers

- **Q^AB/F** - how much of the source stack's edge information survived, 0 to 1.
  The only quality metric available without a ground truth.
- **PSNR** - accuracy against the known perfect result, in decibels. Higher is
  better; +3 dB is a halving of error. Synthetic stacks only.
- **SSIM** - structural similarity, 0 to 1. Synthetic stacks only.
- **Spatial frequency** - how much fine detail is present. A blurred result scores low.
- **Flat-field noise** (pyramid report) - the fused image's high-pass energy over
  the typical frame's, measured only where every source is smooth. Above 1 the
  fusion invented texture there; below it, averaging recovered signal-to-noise.
- **Outside sources** (pyramid report) - the worst pixel, in 8-bit levels, falling
  beyond the range its own frames span at that position. Anything above 0 is a
  value no frame supports, which is what a reconstruction artefact is.
- **Score /100** in the guide - how much of the achievable improvement a method
  captured. 0 means no better than keeping a single frame; 100 means it matched
  the best method in that test.

Timings exclude a discarded warm-up run, so the first configuration is not
charged for imports and model loading. They still come from one machine at one
image size - read them as relative, not absolute.

**Synthetic scenes are not photographs.** Their focus regions are large, smooth
bands, which flatters wide kernels; the default `photographic` style is smooth
gradients with solid shapes, while `texture` is dense noise that is pathological
for variance-based focus measures such as DCT. Tune against your own stack.

---

## Related tests

The claims the guide makes are asserted, not written by hand:

```bash
python -m pytest tests/test_fusion_characteristics.py -v   # the guide's claims
python -m pytest tests/test_fusion_quality.py -v           # the shared quality contract
python -m pytest tests/test_pyramid_flat_field.py -v       # the pyramid report's claims
python -m pytest tests/test_fusion_regression.py -v        # the quality ratchet
python -m pytest tests/ -q                                 # everything
```

Scenes are defined in `tests/fusion_scenarios.py`, metrics in
`tests/fusion_metrics.py`, and the per-method calling conventions and parameter
declarations in `tests/fusion_registry.py`.
