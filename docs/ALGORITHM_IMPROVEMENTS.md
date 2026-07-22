# Fusion algorithms: what could be improved

An audit of the six fusion methods, split into **performance** (speed, memory)
and **quality** (accuracy of the fused result). Every claim here is measured, and
the measurement is given so it can be repeated or disputed.

Measurements were taken on one machine (Python 3.14, OpenCV 5.0, CUDA GPU) with
the synthetic photographic fixture from `tests/synthetic_stack.py`. Read the
absolute timings as relative.

Ranked by expected value: **impact** is how much it changes a real render,
**effort** is rough implementation cost.

---

## Summary

| # | Method | Category | Issue | Impact | Effort |
|---|--------|----------|-------|--------|--------|
| 1 | GFG-FGF | Quality | Focus measured on one colour channel; fails outright when detail is not in it - *fixed in 1.5.7* | High | Low |
| 2 | GFG-FGF | Quality | Whole frames discarded by a global 15% sharpness threshold - *fixed in 1.5.7* | High | Low |
| 3 | DTCWT | Performance | 30% of runtime in two scipy calls that OpenCV does 2-4.5x faster, bit-identically | High | Low |
| 4 | DCT | Quality | Result depends on the order frames are passed in | Medium | Low |
| 5 | IFCNN | Quality | Systematic darkening from truncation instead of rounding - *fixed in 1.5.4* | Medium | Trivial |
| 6 | IFCNN | Quality | Colour drifts through the encode/decode round trip - *fixed in 1.5.5* | Medium | Medium |
| 7 | DTCWT | Quality | Frames fused pairwise and recursively, so the result is order-dependent | Medium | Medium |
| 8 | DTCWT | Performance | CPU cost grows faster than image area | Medium | Medium |
| 9 | Guided Filter | Quality | The exposed kernel parameter barely changes anything - *investigated and closed in 1.5.6* | Low | Low |
| 10 | Guided Filter | Performance | Dead code; per-frame float32 copies dominate memory - *fixed in 1.5.6* | Low | Low |

---

## 1. GFG-FGF measures focus on a single colour channel

**Category: quality. Impact: high. Effort: low. Fixed in 1.5.7.**

`fusion_methods/gfg_fgf.py` picked one channel to work from - whichever had the
largest sum over the first frame - and measured focus only there:

```python
ch_sum = imgs_f32[0].sum(axis=(0, 1))
channel_idx = int(np.argmax(ch_sum))
grays = [img[:, :, channel_idx] for img in imgs_f32]
```

If the subject's detail happens to live in a different channel, the method sees a
flat picture and cannot fuse at all. On a stack whose texture is carried in blue
while red dominates brightness:

| Method | PSNR |
|--------|------|
| Best single unfused frame | 18.41 dB |
| **GFG-FGF** | **18.38 dB** - no better than not fusing |
| Guided Filter | 35.80 dB |
| DCT | 32.93 dB |
| DTCWT | 39.06 dB |

That fixture is deliberately harsh, but the failure mode is real for any subject
whose texture is chromatic rather than tonal - stained biological samples,
printed circuit boards, painted surfaces.

**Fixed in 1.5.7.** Both roles the single channel played were split apart:

- The **guide** for the guided filter is now luminance
  (`cv2.cvtColor(..., COLOR_BGR2GRAY)`), as every other method uses.
- **Activity** - the Scharr focus score and the local-contrast map - is computed
  on all three channels, keeping the strongest response at each pixel. A
  structure that exists in one channel only is therefore still seen at full
  strength rather than being diluted by two flat channels, which is what a
  luminance-only measure would do.

The same split is mirrored in the GPU path (`gfg_fgf_torch.py`), which agrees
with the CPU result to a mean of 0.06 levels.

---

## 2. GFG-FGF discards whole frames

**Category: quality. Impact: high. Effort: low. Fixed in 1.5.7.**

Each frame got one global sharpness score, and any frame below 15% of the best
was dropped entirely (`scale = 0.15`):

```python
if focus_vals[i] < scale * max_focus:
    return i, None
```

A small detailed subject against a plain background pushes the background frame
under the line. Measured on the `depth_edge` scenario, the background frame
scores **12%** of the subject frame - and the output was then one source frame
returned byte-for-byte unchanged, with no fusion performed at all.

**Fixed in 1.5.7.** The intent - skip frames with nothing to contribute - is
sound, but a single global number is the wrong test, because a frame can be
locally the sharpest anywhere while being globally soft. The quota is gone; a
frame is now dropped only when it carries no gradient energy at all
(`focus_vals[i] > 1e-6 * max_focus`, i.e. blank or dead frames), and the
per-pixel decision map decides everything else - which is what the argmax was
always there to do.

### Measured effect of fixes 1 and 2

PSNR against the sharp reference, `HEAD` versus the fix, over the six scenarios
in `tests/fusion_scenarios.py` plus a chromatic fixture whose texture is carried
in blue while red dominates brightness:

| scenario | best single frame | before | after | delta |
|---|---|---|---|---|
| fine_texture | 14.69 | 32.13 | 32.10 | -0.04 |
| sensor_noise | 23.02 | 39.80 | 40.77 | +0.96 |
| depth_edge | 31.88 | 31.88 * | 35.95 | **+4.07** |
| long_stack | 17.59 | 28.37 | 28.59 | +0.22 |
| low_contrast | 45.36 | 60.82 | 61.29 | +0.47 |
| saturated_colour | 23.85 | 29.22 | 31.88 | **+2.66** |
| chromatic_detail | 25.77 | 25.77 * | 40.54 | **+14.77** |

`*` output was a source frame returned unchanged - no fusion at all. Note that
both failures compound on the chromatic fixture: the wrong channel is read, so
every frame scores near zero, so the gate then fires as well.

Cost is one extra colour conversion and three-channel activity instead of one:
0.011 s to 0.015 s on a 320x320 three-frame stack. Covered by
`tests/test_fusion_characteristics.py::test_gfgfgf_keeps_globally_soft_frames`
and `::test_gfgfgf_finds_detail_in_any_colour_channel`.

---

## 3. DTCWT spends 30% of its time in two replaceable scipy calls

**Category: performance. Impact: high. Effort: low.**

Profiling 6 frames at 768x768 (2.56 s total) puts 0.78 s in
`scipy.ndimage.maximum_filter` and `scipy.ndimage.convolve`, both called from
`fuse_highfreq_vectorized`. OpenCV has direct equivalents:

| Operation | scipy | OpenCV | Speed-up | Identical? |
|-----------|-------|--------|----------|------------|
| 3x3 max filter over 6 orientations | 10.09 ms | `cv2.dilate` 2.22 ms | **4.5x** | yes, exactly |
| 3x3 neighbour count | 5.30 ms | `cv2.boxFilter(normalize=False)` 2.61 ms | **2.0x** | yes, exactly |

Verified bit-identical on random input, so this is a drop-in substitution with no
quality trade-off. It also removes the only scipy dependency in the hot path.

**Fix.** Replace the two calls, looping the 6 orientation slices (OpenCV works on
2D planes). Expect roughly a 20-25% cut in DTCWT CPU time.

---

## 4. DCT's result depends on the order frames are given in

**Category: quality. Impact: medium. Effort: low.**

Passing the same stack in a different order should give the same picture. It does
not:

| Method | Agreement across frame orderings |
|--------|----------------------------------|
| StackMFF-V4 | identical |
| Guided Filter | 87.8 dB |
| GFG-FGF | 81.5 dB |
| DTCWT | 46.4 dB |
| **DCT** | **28.1 dB** |

28 dB is a visible difference, not rounding. The cause is `np.argmax` over
per-block variance: in a flat region every frame's variance is near-identical, so
the winner is decided by ties, and `argmax` always breaks a tie toward the
lowest index. Shuffle the inputs and a different frame wins those blocks.

This matters beyond tidiness. Flat regions are exactly where the choice is
arbitrary, so neighbouring blocks can be assigned to different frames for no
reason, which is one source of the blockiness DCT is criticised for.

**Fix.** Break ties deterministically and coherently rather than by index: prefer
the frame already chosen by the neighbouring block, or require a winner to exceed
the runner-up by a small margin before switching. A margin also suppresses the
noise-driven flipping that makes DCT weak on grainy shots.

---

## 5. IFCNN darkens every pixel slightly

**Category: quality. Impact: medium. Effort: trivial.**

`fusion_methods/ifcnn.py:_to_bgr` converts the float result to bytes with
`astype(np.uint8)`, which truncates rather than rounds:

```python
rgb = np.clip(rgb, 0.0, 1.0) * 255.0
return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
```

Truncation removes on average half a level from every pixel. Measured mean shift
from the guided-filter input to the refined output, per channel:

```
B -0.43   G -0.81   R -0.21      (average -0.48, matching the -0.5 predicted)
```

**Fix.** `np.rint(rgb).astype(np.uint8)`, or `(rgb + 0.5)`. One line. It will not
fix the whole colour drift (see next item) but it removes a bias that is pure
loss. The same pattern is worth checking anywhere else a float result is cast.

**Fixed in 1.5.4.** Both truncating casts in `ifcnn.py` now round: `_to_bgr` and
the tile accumulator in `_refine_tiled`, which had the same bias. A normalize /
denormalize round trip through the tensor conversion helpers is now exact (max
error 0 levels, was 1).
Mean shift from the guided-filter input to the refined output, across the six
characterisation scenarios:

| scenario | before | after | delta |
|---|---|---|---|
| fine_texture | -1.043 | -0.584 | +0.459 |
| sensor_noise | -0.132 | +0.367 | +0.499 |
| depth_edge | -0.994 | -0.508 | +0.485 |
| long_stack | -0.090 | +0.398 | +0.489 |
| low_contrast | -0.036 | +0.457 | +0.493 |
| saturated_colour | -3.837 | -3.404 | +0.433 |
| **average** | **-1.022** | **-0.546** | **+0.476** |

The recovered +0.476 matches the predicted +0.5. The residual shift is the
round-trip colour drift of item 6, which is a separate cause.

### The same pattern elsewhere

The closing note above was followed up: every float-to-`uint8` cast in
`fusion_methods/` and `core/` was audited. Nine candidates, six genuine:

| site | verdict |
|---|---|
| `dtcwt.py:165` | truncated a reconstructed float image - fixed |
| `gff.py:250` | truncated a reconstructed float image - fixed |
| `gfg_fgf.py:313` | truncated each float channel - fixed |
| `multi_focus_fusion.py:787` | truncated the tiled weighted accumulator - fixed |
| `multi_focus_fusion.py:872` | truncated the tiled weighted accumulator - fixed |
| `workers.py:863` | truncated non-`uint8` frames on GIF export - fixed |
| `stackmffv4.py:203`, `:302` | no-op: `fused_color` is a fancy-index gather from a `uint8` stack, so nothing is truncated |
| `registration.py:702` | no-op: `map_coordinates` with a `uint8` input returns `uint8` and rounds internally, matching the `warpPerspective` CPU path |
| `dct.py:153` | not a pixel cast - an index map, where rounding would be wrong |

Effect on the classical methods, measured against `HEAD` on the same six
scenarios (mean shift in levels, PSNR in dB against the sharp reference):

| method | mean shift | avg dPSNR | best case |
|---|---|---|---|
| DTCWT | +0.476 | +2.41 | low_contrast +13.50 |
| Guided Filter | +0.418 | +2.15 | low_contrast +12.49 |
| GFG-FGF | +0.100 | +0.51 | low_contrast +3.02 |

The low-contrast scenario dominates because when the whole frame sits in a narrow
band of levels, a systematic half-level bias *is* most of the error. GFG-FGF
gains least because its division already lands many pixels on exact integers.

---

## 6. IFCNN's colour drifts through the round trip

**Category: quality. Impact: medium. Effort: medium.**

The refinement stage encodes the picture into the network's feature space and
decodes it back. That round trip is lossy, and the loss shows up as colour:
IFCNN's colour deviation averages about **6x** the guided filter's across the six
characterisation scenarios, and on one of them the refined picture scores *below*
simply keeping the sharpest single frame.

The stage does do its job - it improves boundary rendering in 4 of 6 scenarios.
The problem is that it pays for that with a whole-frame colour cost.

A cheap experiment supports separating the two. Taking only IFCNN's **luminance**
(where the edge repair lives) and keeping the original result's **chroma**:

| | PSNR | Colour error |
|---|------|--------------|
| Guided Filter | 39.48 dB | 0.79 |
| + IFCNN as it stands | 30.20 dB | 4.66 |
| + IFCNN, original chroma kept | **33.42 dB** | **3.06** |

That recovers 3.2 dB for a few lines of LAB conversion. It does not close the gap
to the unrefined result, which says the remaining loss is in luminance too - so
the fuller fix is to blend IFCNN's output only where it actually disagrees with
the input near edges, rather than adopting it everywhere.

**Worth noting:** these are synthetic stacks, which give IFCNN no genuinely
missed detail to recover, so they show its costs and not its benefits. Confirm
against a real stack before changing behaviour.

**Fixed in 1.5.5**, by a route that turned out cleaner than either idea above.

The round trip is lossy, but the loss is *measurable in isolation*: decode the
candidate's own features, with nothing merged in, and the difference from the
candidate is the round-trip error by itself, carrying no fusion information. So
`_refine_block` now returns

```
candidate + (decode(merged features) - decode(candidate's features))
```

instead of `decode(merged features)`. The bracket is what merging the sources
contributed; the drift cancels. Where IFCNN finds nothing to add, the candidate
comes back untouched rather than paying a whole-frame round trip for nothing -
refine an image against itself and the output is now bit-identical to the input.

No thresholds, no edge mask, no colour-space conversion. The cost is one extra
decode per block (conv3 + conv4 on a feature map that is already computed);
encoding, which dominates, is unchanged.

Measured over the six characterisation scenarios, against the guided filter it
refines:

| | before | after | guided filter |
|---|--------|-------|---------------|
| PSNR | 30.14 dB | **37.14 dB** | 39.70 dB |
| Edge PSNR | 29.16 dB | **33.37 dB** | 29.95 dB |
| Colour error | 6.89 | **2.87** | 1.18 |

Both of the report's warnings about the stage are now obsolete. It no longer
scores below the best single frame anywhere (it did on `depth_edge`), and it now
improves boundary rendering in 6 of 6 scenarios rather than 4. It still costs
overall PSNR on 3 of 6 - these stacks hand it a near-perfect fusion and no missed
detail, so there is nothing to win and something to disturb - but the worst case
is 8.5 dB behind rather than 18.

An edge-gated variant was measured alongside it: same residual, scaled by a
blurred Laplacian of the candidate. It reads better on whole-frame numbers
(PSNR 38.60, colour 2.27) because it suppresses the correction over most of the
picture, but it gives back 1.3 dB at boundaries - which is the one thing the
stage exists to improve - in exchange for two tuned constants. Not taken.

---

## 7. DTCWT fuses frames pairwise, so order matters

**Category: quality. Impact: medium. Effort: medium.**

With more than two frames, `fuse_highfreq_vectorized` folds them together two at
a time:

```python
if num_imgs > 2:
    fused = coeffs_list[0]
    for i in range(1, num_imgs):
        fused = fuse_highfreq_vectorized([fused, coeffs_list[i]], window_size)
```

Fusing `(((1,2),3),4)` is not the same as choosing the best of all four at once:
early frames pass through more rounds of the consistency filter than late ones.
Measured order-agreement is 46.4 dB (see item 4's table), and the cost grows with
stack length - which is precisely the macro use case.

**Fix.** Select across all frames jointly - compute the activity measure for every
frame, then take the winner per coefficient in one pass, as the other methods do.
That is also faster: one pass instead of N-1.

---

## 8. DTCWT's CPU cost grows faster than image area

**Category: performance. Impact: medium. Effort: medium.**

Six frames, CPU against GPU:

| Size | DTCWT CPU | DTCWT GPU | Guided Filter CPU | DCT CPU |
|------|-----------|-----------|-------------------|---------|
| 512x512 | 1.11 s | 0.15 s (7.5x) | 0.06 s | 0.01 s |
| 1024x1024 | 5.09 s | 0.41 s (12.3x) | 0.33 s | 0.08 s |
| 2048x2048 | **29.81 s** | 0.22 s (138x) | 1.43 s | 0.36 s |

Quadrupling the pixels multiplies CPU time by 4.6x then 5.9x, so the cost is
growing faster than area - consistent with the pairwise recursion in item 7 and
with memory pressure from holding every frame's full coefficient pyramid.

At 2048x2048 the CPU path takes half a minute while the GPU path takes a fifth of
a second. Fixing items 3 and 7 should bring the CPU path down substantially; in
the meantime, this is the strongest argument for making the GPU path the default
whenever `pytorch_wavelets` and `pywt` are installed.

---

## 9. The Guided Filter's kernel parameter does almost nothing

**Category: quality. Impact: low. Effort: low.**

The kernel slider is exposed in the UI, so users reasonably assume it matters.
Swept across a 13-fold range:

| kernel_size                 | 7     | 15    | 31    | 63    | 95    |
|-----------------------------|-------|-------|-------|-------|-------|
| PSNR (as first measured)    | 41.78 | 42.27 | 42.27 | 42.31 | 42.32 |
| PSNR (re-measured at 1.5.5) | 42.50 | 43.12 | 43.23 | 43.24 | 43.24 |

Under 0.8 dB from end to end, and flat above 15. For comparison, DTCWT's `N`
moves its result by 13.6 dB and GFG-FGF's kernel by 5.9 dB.

**Resolved in 1.5.6: the first reading is correct, and structurally so.** The
second reading - that the base/detail split is contributing less than intended -
was tested and rejected. GFF reconstructs `base + detail` exactly, so with
normalised weights the output is

```text
fused = Σ wd_k·I_k  +  Σ (wb_k − wd_k)·B_k
```

The base layers `B_k` - the only thing the kernel controls - reach the output
solely through the *gap* between the two weight maps. That gap is small, so the
kernel is near-inert by construction, not by accident.

Measured directly, comparing each setting's output against the default's rather
than against the reference (higher dB = more alike; 60 dB is roughly 0.16 grey
levels RMS):

| kernel_size          | 7    | 15   | 31 | 63   | 95   |
|----------------------|------|------|----|------|------|
| PSNR vs. k=31 output | 50.9 | 58.0 | -  | 68.0 | 67.6 |

Every setting is visually identical to every other. The same holds for `r1`, the
base-layer weight radius the kernel feeds: a 21-fold sweep (7 → 150) moves the
output by 60-70 dB. The whole base-layer branch is inert.

Two changes followed:

- `r1` is now derived from the kernel at the paper's own 3:1 ratio
  (`base_weight_radius()`), instead of being pinned at 45. `kernel_size=31`
  still yields exactly `r1=45`, so the default is unchanged, but the weight
  smoothing now tracks the decomposition scale. That removes the one setting
  where the slider actively hurt: small kernels used to degrade the result
  (42.69 dB at k=7 against 43.42 at the default), and now do not. Sweep spread
  falls from 0.74 dB to 0.03 dB - the control is flat *and* safe everywhere.
- The method help no longer promises the slider will "balance sharpness and
  smoothness". It says the setting has almost no visible effect and explains why.

The slider itself is left in place and full-range: it is shared with DCT and
GFG-FGF, where it does matter, and every value is now harmless.

---

## 10. Guided Filter: dead code and memory headroom

**Category: performance. Impact: low. Effort: low.**

- `fusion_methods/gff.py` defines a module-level `guided_filter()` that nothing
  calls; the real work happens in the nested `run_gf`. Deleting it removes a
  misleading second implementation that could drift from the one in use.
- The stack is converted to float32 BGR up front (`stack_flt`), which is 12 bytes
  per pixel per frame - 2.9 GB for ten 24-megapixel frames, before any working
  buffers. Tiling exists for this reason, but converting lazily per frame, or
  keeping the guide as single-channel float and the sources as uint8 until they
  are needed, would raise the threshold at which tiling becomes necessary.
- Weights are accumulated in thread-completion order, so results vary by one
  level between runs. Harmless numerically, but it means output is not
  reproducible bit-for-bit; accumulating in a fixed order would cost nothing
  measurable.

**Fixed in 1.5.6.** The dead `guided_filter()` is gone. The three full-stack
float32 lists (`stack_flt`, `base_layers`, `detail_layers`) and the stacked
saliency maps are gone with it: frames are converted on demand, saliency is
reduced with a running maximum, and decomposition now happens inside the fusion
pass. `_map_in_order()` keeps a sliding window of exactly `max_workers` tasks
outstanding, so peak memory is set by the pool size rather than by stack depth,
and results are consumed in index order - which also makes accumulation
reproducible, so `deterministic=False` has been dropped from the registry.

Peak working set, 1536x1536 frames:

| frames | 4       | 8        | 16       | 24       |
|--------|---------|----------|----------|----------|
| before | 962 MiB | 1889 MiB | 3603 MiB | 5570 MiB |
| after  | 891 MiB | 1772 MiB | 1722 MiB | 1704 MiB |

Growth with stack depth stops once the depth exceeds the worker count; the curve
is flat from there. Cost is ~5% throughput (0.99 s → 1.04 s on 16 frames), from
capping the default pool at 8 threads - the OpenCV filters are internally
parallel, so the extra Python threads bought little. Output is bit-identical to
the previous implementation on the test stack.

---

## Cross-cutting

**GPU is consistently worth it, and is not the default.** Speed-ups measured
above are 2.6-8.6x for the guided filter and DCT, and 7.5-138x for DTCWT. The
GPU paths are already implemented and verified to agree with their CPU twins by
`tests/test_fusion_quality.py::test_gpu_matches_cpu`.

**No method reports its confidence.** Every one produces a decision map it then
throws away. Surfacing "how sure was it here" - the margin between the best and
second-best frame - would let the UI flag regions where the stack simply lacks a
sharp frame, which is the most common cause of a disappointing render and is
currently invisible to the user.

**The neural methods have no CPU fallback path worth using.** StackMFF-V4 on CPU
is usable but slow, and there is no smaller variant. Not a defect, but it shapes
the advice given to users without a GPU.

---

## Reproducing these measurements

```bash
python -m pytest tests/ -q                                     # the whole suite
python tests/benchmark_fusion_quality.py --synthetic --all     # cross-method table
python tests/benchmark_fusion_quality.py --synthetic --method guided_filter --param kernel_size
python tests/visualize_fusion_characteristics.py --open        # the six scenarios
```

Scenario definitions are in `tests/fusion_scenarios.py`, metrics in
`tests/fusion_metrics.py`, and the per-method calling conventions in
`tests/fusion_registry.py`.
