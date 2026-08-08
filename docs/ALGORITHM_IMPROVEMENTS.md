# Fusion algorithms: what could be improved

An audit of the six fusion methods, split into **performance** (speed, memory)
and **quality** (accuracy of the fused result). Every claim here is measured, and
the measurement is given so it can be repeated or disputed.

The alignment stages that run *before* fusion are audited separately, to the same
standard, in [REGISTRATION_IMPROVEMENTS.md](REGISTRATION_IMPROVEMENTS.md).

Measurements were taken on one machine (Python 3.14, OpenCV 5.0, CUDA GPU) with
the synthetic photographic fixture from `tests/synthetic_stack.py`. Read the
absolute timings as relative.

Ranked by expected value: **impact** is how much it changes a real render,
**effort** is rough implementation cost.

Items 1-10 are the original audit. Items 11-16 were added by a follow-up at
1.11.0 that re-verified every claim below against the code; the follow-up also
covered the two methods added since the audit - `pyramid.py` and `depthmap.py` -
which were written to this document's standards and contribute only items 15
and 16. Items 17-19 came from real stacks rather than from an audit, and all
three are the same failure in three methods' clothing: a selection rule with no
answer for the parts of a frame where nothing is in focus. Item 20 is the one
performance gap left by the two methods added after the original audit. Items
21-24 also came from real stacks, and 24 is the same shape as 18: a rule that
works on a test fixture and quietly stops working as the stack gets deeper. The
**Fixed** column
tracks how much of each item has actually landed; a partial percentage means a
mitigation shipped but the underlying issue remains.

---

## Summary

| # | Method | Category | Issue | Impact | Effort | Fixed |
|---|--------|----------|-------|--------|--------|-------|
| 1 | GFG-FGF | Quality | Focus measured on one colour channel; fails outright when detail is not in it - *fixed in 1.5.7* | High | Low | 100% |
| 2 | GFG-FGF | Quality | Whole frames discarded by a global 15% sharpness threshold - *fixed in 1.5.7* | High | Low | 100% |
| 3 | DTCWT | Performance | 30% of runtime in two scipy calls that OpenCV does 2-4.5x faster, bit-identically - *fixed in 1.11.1* | High | Low | 100% |
| 4 | DCT | Quality | Result depends on the order frames are passed in - *fixed in 1.30.12* | Medium | Low | 100% |
| 5 | IFCNN | Quality | Systematic darkening from truncation instead of rounding - *fixed in 1.5.4* | Medium | Trivial | 100% |
| 6 | IFCNN | Quality | Colour drifts through the encode/decode round trip - *fixed in 1.5.5* | Medium | Medium | 100% |
| 7 | DTCWT | Quality | Frames fused pairwise and recursively, so the result is order-dependent - *fixed in 1.30.13* | Medium | Medium | 100% |
| 8 | DTCWT | Performance | CPU cost grows faster than image area | Medium | Medium | 25% |
| 9 | Guided Filter | Quality | The exposed kernel parameter barely changes anything - *investigated and closed in 1.5.6* | Low | Low | 100% |
| 10 | Guided Filter | Performance | Dead code; per-frame float32 copies dominate memory - *fixed in 1.5.6* | Low | Low | 100% |
| 11 | DCT | Quality | Crashes (default kernel) or corrupts indices on stacks of 256+ frames - *fixed in 1.11.2* | High | Low | 100% |
| 12 | DTCWT | Quality | Each colour channel picks its own source frame, so colour splits at depth edges - *fixed in 1.14.0 and 1.30.14* | Medium | Low | 100% |
| 13 | DTCWT, Pyramid, Depth Map | Robustness | Folder loader copy-pasted four ways; mixed filenames crash the sort | Low | Low | 0% |
| 14 | DTCWT | Quality | Consistency vote biased toward the later frame at image borders - *dissolved with the vote in 1.30.13* | Low | Trivial | 100% |
| 15 | Depth Map | Quality | MODE_MAX decision map has no regularisation, so near-tie seams can speckle | Low | Low | 0% |
| 16 | DTCWT, Pyramid | Quality | Lowpass/base band fused by plain mean; ghosts under exposure drift - *fixed in 1.11.3* | Low | Medium | 100% |
| 17 | DCT | Quality | Focus measured as total block contrast, and grain deciding the rest, so blurred frames blank whole regions - *fixed in 1.15.1 and 1.15.2* | High | Low | 100% |
| 18 | DCT | Quality | Per-block winner never clears its margin in a deep stack, so 91% of blocks fell to a fallback that tore the background - *fixed in 1.17.1* | High | Medium | 100% |
| 19 | Pyramid | Quality | Choose-max stitches defocused regions out of frames that disagree, reconstructing filaments no frame had, and loses them to grain - *fixed in 1.19.0* | High | Medium | 100% |
| 20 | Depth Map | Performance | No device path at all, and a measurement pool bounded by thread count rather than by memory - *fixed in 1.31.0* | Medium | Medium | 100% |
| 21 | Depth Map | Quality | Box pooling is edge-blind, so a sharp contour claims the background beside it and rings every subject - *fixed in 1.33.0* | High | Low | 100% |
| 22 | Depth Map | Quality | The argmax has no spatial prior, so a region nothing resolves tears into confetti - *fixed in 1.34.0* | High | Low | 100% |
| 23 | Depth Map (Max) | Quality | Still hard-selects where the measurement supports no selection, leaving item 22's confetti as coarse patches - *fixed in 1.35.0, corrected in 1.36.0* | High | Medium | 100% |
| 24 | Depth Map (Average) | Quality | Linear contrast weighting stops selecting as the stack deepens, so the blend collapses into the plain mean of every frame and hazes over - *fixed in 1.37.0* | High | Low | 100% |
| 24b | Depth Map | Quality | Halo radius unbounded by the pooling window, so a radius meant for a wide kernel fills a band around every contour with defocus - *ranged in 1.37.0* | Medium | Trivial | 100% |
| 25 | Depth Map (Average) | Quality | The blend selects hard once it selects at all, and had nothing to make that selection coherent, so a region no frame resolves broke into blotches - *fixed in 1.38.0* | High | Medium | 100% |
| 26 | Depth Map (Average) | Quality | No coherence along the frame axis, so the busiest fifth of the frame stayed 18% above the reference at every spatial setting - *fixed in 1.39.0* | Medium | Medium | 100% |
| 27 | Depth Map (Max) | Quality | Every pixel comes from one frame, so no dial could divide its grain and the residual against the reference was pinned - *fixed in 1.42.0* | Medium | Low | 100% |
| 28 | Depth Map (Max) | Quality | No edge-aware stage over the depth field, so tile agreement stalled at 0.96 in the mid-detail regions the trust gate declines to touch - *fixed in 1.43.0* | Medium | Low | 100% |

Row 24b is numbered that way because it is a sub-finding of item 24 and is
documented inside it; every other row matches its `## N` section below.

**Overall: 90% done** - 26 of 29 rows fully fixed, item 8 partially (the GPU
default shipped; the CPU cost itself is untouched, and 1.30.13 showed the
pairwise fold was not what made it grow), 2 untouched. Since 1.17.1 a
quality ratchet (`tests/test_fusion_regression.py`, described after item 19)
guards every method against silent regressions of the kind items 17, 18 and 19
were.

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

**Category: performance. Impact: high. Effort: low. Fixed in 1.11.1.**

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

**Fixed in 1.11.1**, exactly as prescribed: both calls replaced, looping the
six orientation slices. `cv2.dilate` stands in for the max filter - its
default border ignores out-of-bounds pixels, which for a max filter is the
same set of values scipy's reflect mode produces, since reflection only
duplicates pixels already inside the window. The neighbour count uses
`cv2.boxFilter(normalize=False, borderType=BORDER_CONSTANT)`, whose
zero-padding matches the previous `mode='constant', cval=0.0` - which keeps
the item 14 border bias, deliberately, so the swap stays bit-identical (see
the note there).

Verified bit-identical against the scipy path at the filter level (random
complex input, sizes 7x129 to 768x768, windows 3/5/7) and end to end: the
fused image from a 6-frame 768x768 photographic stack is byte-for-byte the
same as before the change. DTCWT CPU time on that stack fell from 1.53 s to
1.16 s (best of 3) - a 24.5% cut, within the predicted 20-25%. The scipy
import is gone from `dtcwt.py`, and with it the last direct scipy import in
the codebase; the registry check and the install hint in
`core/multi_focus_fusion.py` no longer ask for scipy. (scipy is still listed
in `requirements.txt` and the PyInstaller build commands - dropping it from
packaging is a separate decision.)

---

## 4. DCT's result depends on the order frames are given in

**Category: quality. Impact: medium. Effort: low. Fixed in 1.30.12.**

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

**Status at 1.11.0: open.** The implementation has since moved from `np.argmax`
to a running maximum (`var_map > max_variance_map`), which keeps the identical
lowest-index tie bias, and the GPU twin reproduces it deliberately
(`dct_torch.py`: "preserves first-max-wins tie behavior of the CPU path"). Any
fix must land in both paths together to keep `test_gpu_matches_cpu` green.

### The question changed at 1.17.1

Item 18 replaced "which frame wins this block" with "where along the stack is
this block's focal plane". The plane is a *position in the input order*, so
after that rewrite the order stopped being an accident the method should be
insensitive to and became part of what it models - and the numbers went the
wrong way, 28.1 dB at the audit and 25.1 dB on `deep_stack` by 1.30.11.

Full invariance and item 17's fix cannot both be had. In a region no frame
resolves, every frame is equally poor; deciding by intrinsic energy is exactly
the noise contest item 17 removed, and averaging all of them is a blur. Item 18
answered "mid-stack", which is stable, uniform and unavoidably a statement about
the order. So the requirement had to be restated as two that can be met:

- **Reversing the stack must render the identical picture.** Back-to-front is
  still the same sweep, so this one is an equality, not a tolerance.
- **The plane must lie in the run of frames the block is actually in focus at.**
  This is what was broken, independently of any re-ordering.

**The plane was a mean over frames with nothing to do with each other.** It was
the mean index of *every* frame clearing the plateau bar, wherever in the stack
they sat. Two frames at opposite ends both clearing it - one because it resolves
the block, one on grain - put the plane halfway between, on a frame that
resolves nothing. Measured on `deep_stack`, 43% of blocks had an in-focus set
that was not one run. Restricted to blocks where a frame genuinely is in focus
(peak at least twice the block's median energy), 0.13% of them were given a
plane more than half a frame from any in-focus frame, the worst 4.5 frames away;
on the veil fixture 0.46% and 4.5. Small shares, but each one is a block
rendered from a frame that is blurred there.

**Fixed in 1.30.12,** in two parts, and they turn out to fix different things.

- **Read the in-focus set as a run** (`_focal_plane` in `dct.py`). The plane is
  the middle of the run containing the peak, so frames elsewhere in the stack
  cannot enter the average at all. A run ends where a frame falls to `_RUN_EXIT`
  (0.75) of the in-focus bar rather than at the first frame below the bar
  itself: where nothing is in focus every frame sits within a few percent of
  every other, so a run that ends at the first dip collapses onto whichever
  frame grain favoured - which is item 17's veil artefact, 71% of the veil
  fixture's smooth body. Hysteresis is the right shape because it asks how deep
  a dip is, not how long: grain moves a defocused frame a few percent, leaving
  the depth of field costs it most of its energy. Only frames above the upper
  bar extend the run's ends, so the lower bar lengthens no plateau.
- **Carry the plane at half-frame resolution** (`_SUBFRAME`). Item 18 estimates
  the plane to sub-frame accuracy and 1.30.11 then rounded it to a whole frame
  before the median filter, throwing that away. The map is now held in halves
  of a frame - the exact resolution a run's midpoint needs - through the filter
  and into compositing.

Measured on every fixture the project has:

| scenario | PSNR before | after | shuffle before | after | reversed before | after |
|---|---|---|---|---|---|---|
| fine_texture | 33.59 | 33.89 | 45.63 | 47.97 | exact | exact |
| sensor_noise | 38.18 | 41.77 | 38.75 | 45.96 | exact | exact |
| depth_edge | 34.39 | 35.27 | 40.93 | **exact** | 40.93 | **exact** |
| long_stack | 28.32 | **30.54** | 25.39 | **30.48** | 26.49 | **exact** |
| low_contrast | 61.00 | 63.15 | 63.49 | 66.76 | exact | exact |
| saturated_colour | 29.92 | 29.74 | 34.20 | 37.05 | exact | 43.48 |
| deep_stack | 32.61 | 32.17 | 25.12 | **34.35** | 47.49 | **exact** |
| wash | 29.79 | 30.27 | 40.49 | **exact** | 40.49 | **exact** |
| veil | 34.56 | 34.71 | 21.98 | **29.36** | 44.20 | **exact** |

"Shuffle" is the mean agreement of four random orderings with the stack as
given; "reversed" is the reversed stack against it, where *exact* means
byte-for-byte. Reversal is now exact on eight of the nine. `saturated_colour` is
the exception and the reason is a genuine ambiguity rather than a rounding one:
102 of its 1600 blocks are rendered identically by frames 0 and 2 with frame 1
blank between them, so the block holds two disjoint runs whose peaks are equal
to the last float bit and the first to arrive wins. Merging tied runs would put
the plane on the blank frame between them, which is worse than either answer.

**The two parts fix different halves, which is worth stating separately** since
either could be mistaken for the other's work:

| | 1.30.11 | + half-frame plane | + run rule |
|---|---|---|---|
| PSNR, sensor_noise | 38.18 | **41.77** | 41.77 |
| PSNR, long_stack | 28.32 | **30.54** | 30.54 |
| shuffle, deep_stack | 25.12 | 25.76 | **34.35** |
| shuffle, veil | 21.98 | 24.02 | **29.36** |

Every PSNR gain is the half-frame plane; every order-agreement gain is the run
rule. On blocks where a frame is genuinely in focus the plane is now within half
a frame of one on all nine fixtures - 0.00%, against the 0.13% and 0.46% above.

**What it costs.** `deep_stack` loses 0.43 dB and `saturated_colour` 0.19. The
first is the run rule reaching further into a background no frame resolves,
which renders it from nearer mid-stack: the softness half of the trade item 18
describes, on the fixture whose reference declares the least-blurred rendering
correct. Two ratchet metrics moved past tolerance and were re-recorded
deliberately - `long_stack/defocus_seam_excess` 10.79 to 11.22, on a fixture
where DCT still leads the pyramid's 13.62 and DTCWT's 13.39, and
`deep_stack/block_speckle` by 0.156, which is one 8x8 cell out of 640, the
metric's own quantum.

**`sensor_noise` gained 3.59 dB and it is not a sharper measure.** Compositing
two neighbouring frames averages their independent grain, which is worth about
3 dB, and that is enough to carry DCT past the guided filter (41.6) and GFG-FGF
(40.8) on that fixture. With compositing off the measure alone scores 36.7,
below all three. `test_dct_struggles_most_with_noise` now asserts the two
separately, because "DCT is the weakest of the classical methods on a noisy
stack" is only still true of its selection, not of what it renders.

**Cost: none measurable.** The run tracking is a handful of operations on the
block lattice, and both passes over the frames were already there: 64 frames of
1024x1024 run 838 ms before and 820 ms after, 24 frames of 2048x2048 1387 ms
and 1391 ms.

Both paths landed together, as item 11 requires - `dct_torch.py` mirrors the run
tracking on the device and the plane still crosses to the host for the median.
CPU/GPU agreement on the veil fixture *improved*, from the 41.5 dB item 17
recorded to 66.2 dB: the two paths estimate their noise floors slightly
differently, and a run is less sensitive to that than a single winner is.

**Guarded by** `tests/test_dct_frame_order.py`, which drives `_focal_plane`
directly for each situation it has to get right and asserts the reversal
equality end to end. Every one of its checks fails on the 1.30.11 rule.

**What is still order-dependent, and stays that way.** A shuffled stack is not a
sweep, so the plane cannot mean what it normally does and the pictures still
differ - most in regions no frame resolves, which is where "mid-stack" is the
only answer and mid-stack is a different frame once the order changes. Closing
that would need the sweep order recovered from the images themselves, which is a
different piece of work; the method now states the assumption in its docstring
instead of leaving it implicit.

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

**Category: quality. Impact: medium. Effort: medium. Fixed in 1.30.13.**

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

**Status at 1.11.0: open.** `dtcwt_torch.py` replicates the pairwise order
deliberately ("matches the sequential pairwise fusion order exactly"), so the
joint-selection rewrite must change both paths in the same commit.

### Fixed in 1.30.13 - and the consistency vote had to go with the chain

The selection is now the one this item asks for. Every frame's coefficients are
scored once, on their own, and each coefficient goes to whichever frame carries
the most activity around it (`_coefficient_activity` and `_keep_stronger` in
`dtcwt.py`, mirrored in `dtcwt_torch.py`). The pass still streams a frame at a
time, so memory is still bounded by the running winner rather than by stack
depth, but the fold is a running maximum instead of a chain: no frame passes
through more rounds of anything than any other.

**The consistency vote could not come along, and that is the whole of the
design decision here.** It is binary by construction - count how many of a
coefficient's neighbours preferred source 1, compare against half the window -
and there is no first source to count when the comparison is against the whole
stack. Two ways out were measured:

- **Generalise it to a mode filter.** Keep the winning *index* per coefficient,
  take the majority index over each 3x3 window, then gather. It is the faithful
  N-ary reading of the vote, and it costs a second pass over the frames -
  a streaming pass holds the winner, not the stack, so the coefficients a
  corrected index points at are no longer in memory and every frame has to be
  transformed again. Measured on the fixtures: 1.7-1.8x the CPU time of the
  rule that shipped, for **worse** order agreement than it (56.2 dB on `deep_stack`
  and 57.0 on the veil fixture, against byte-exact), because a mode filter has
  ties of its own and the only thing left to break them with is the frame
  index.
- **Pool the activity before comparing, and drop the vote.** The vote existed
  to stop a lone coefficient deciding for itself; pooling the activity over the
  same window it used to count over does that in the same pass, one filter per
  frame. This is what `pyramid.py` (`_band_energy`) and `dct.py` since item 17
  already do, and it is what shipped.

So a frame's claim on a coefficient is now Lewis's activity - the maximum
filter over a 3x3 window of the strongest channel's magnitude, unchanged -
pooled over the same window, and the frames are compared on that. Item 12's
shared decision survives it: the channels are still reduced with a maximum
before scoring, so one decision selects all three and a pixel's colour cannot
split across frames.

**Ties are settled on the coefficient rather than on arrival.** Exact ties are
not rare here: a strong structure two frames share fills the maximum filter for
every coefficient around it, so their pooled scores come out equal where their
own coefficients are not - 1.55% of `long_stack`'s finest level, with the tied
frames disagreeing by up to 94% of the level's peak magnitude. Taking the
larger magnitude at the coefficient itself settles those; taking whichever
frame arrived first would have put the order dependence straight back (55.3 dB
against 64.2 on `long_stack`).

**What it is worth.** Every fixture the project has, CPU path, PSNR against
each fixture's own reference:

| scenario | before | after | delta |
|---|---|---|---|
| fine_texture | 37.66 | **38.79** | +1.13 |
| sensor_noise | 44.20 | **45.14** | +0.94 |
| depth_edge | 37.42 | 37.31 | -0.11 |
| long_stack | 32.65 | **35.46** | **+2.81** |
| low_contrast | 64.42 | **65.00** | +0.58 |
| saturated_colour | 33.49 | **33.65** | +0.16 |
| deep_stack | 26.01 | 26.02 | +0.01 |
| wash | 31.71 | 31.80 | +0.09 |
| veil | 15.24 | 15.24 | 0.00 |

The gain is largest where the chain was longest, which is the shape the defect
predicts: `long_stack` is twelve frames, and its first frame used to pass
through eleven consistency filters while its last passed through one. It is
enough to move the method past StackMFF-V4 on that scenario (34.06 dB), which
`tests/test_fusion_characteristics.py` had recorded as the neural model's.

**And the order dependence it was written for.** "Reversed" is the reversed
stack against the stack as given, "shuffles" four random orderings against it;
*exact* means byte for byte:

| scenario | reversed, before | after | shuffles, before | after |
|---|---|---|---|---|
| fine_texture | 42.28 | **exact** | 42.3-47.8 | **exact** |
| sensor_noise | 53.85 | **exact** | 53.9-59.0 | **exact** |
| depth_edge | 71.26 | 103.01 | 71.3-exact | 103.0-exact |
| long_stack | 44.22 | **64.20** | 45.1-48.6 | **64.2-exact** |
| low_contrast | 73.01 | **exact** | 73.0-79.3 | **exact** |
| saturated_colour | 44.48 | **exact** | 44.5-50.5 | **exact** |
| deep_stack | 46.49 | **exact** | 47.5-47.8 | **exact** |
| wash | 59.61 | **exact** | 59.6-exact | **exact** |
| veil | 46.35 | **exact** | 46.8-48.6 | **exact** |

Seven of the nine are now byte-identical however the stack is handed over, and
the GPU path lands the same way. What is left is the tie above where the
magnitudes tie as well: `long_stack` moves 334 pixels of 102,400 by up to 18
levels and `depth_edge` one pixel by one. Both are coefficients where two
frames agree on their neighbourhood and on their own strength and differ only
in phase, and the next tie-break after that would be an arbitrary rule about
complex numbers rather than a statement about focus, so they stay.

**One line of it is not the selection at all.** The lowpass band is an
activity-weighted mean (item 16), and a float sum has a value that depends on
the order it is accumulated in. Left in float32 it costs the exactness above on
four fixtures - 4 to 7 pixels by one level, 96-103 dB - so the two running sums
are now carried in float64 and converted back once at the end. The band is
1/256 of the pixels at the default four levels, so this is free; the GPU path
does the same wherever the device has float64, which is everywhere except MPS.

**Item 14 is dissolved by this, not fixed.** Its border bias was the fixed
`window^2 / 2` threshold meeting a zero-padded neighbour count, and there is no
threshold and no count any more; both filters that remain treat the frame edge
the same way for every frame. Measured on pairs of frames with identical
statistics and no real difference in sharpness, where the honest answer is half
each - the share of coefficients taken from the first frame:

| | border ring | interior |
|---|---|---|
| the pairwise vote | 27.0-29.0% | 49.2-50.5% |
| now | 49.6-52.5% | 49.2-50.3% |

**Cost: it is slightly cheaper, as predicted, and it does not touch item 8.**
One scored pass over the stack replaces N-1 pairwise fusions, which is two
filters per frame instead of three per pair. Old against new on the same
machine in one process, order-balanced so neither implementation gains from
running first, best of six:

| stack | before | after |
|---|---|---|
| 6x 768x768 | 2.23 s | 2.04 s |
| 12x 768x768 | 4.14 s | 3.87 s |
| 24x 512x512 | 3.85 s | 3.26 s |
| 6x 1024x1024 | 4.61 s | 4.15 s |

6 to 15% off the whole method, and the GPU path is unchanged (0.96-1.00x over
6 to 64 frames). Item 8 guessed that the pairwise recursion was behind the CPU
cost growing faster than area; it is not. Quadrupling the pixels at 6 frames
multiplies the time by 5.5x then 5.3x before and 5.7x then 5.4x after, so the
growth is in the transform and its memory traffic, and item 8 keeps its 25%.

**What was measured and not taken.** Dropping the maximum filter - comparing
the pooled magnitude directly, without Lewis's activity step - scored better on
five fixtures (up to +0.45 dB on `sensor_noise`) and worse on two (-0.19 dB on
`depth_edge`), and agreed with itself slightly better on `long_stack`. That is
a change to the *measure*, where this item is about the *selection*, and it is
not a clear enough win to make both in one commit and be unable to say which
did what. It is worth its own item.

**Guarded by** `tests/test_dtcwt_frame_order.py`, which pins the rule at the
level of its two helpers - including what a tie does, and that folding the same
frames in any order lands in the same place - and then asserts the reversal
equality end to end on both paths. On the pairwise rule, reversing a 5-frame
photographic stack moves 24,081 of its 36,864 pixels by up to 153 levels, and
the 60 dB floor the scenario tests use fails on four of the six.

**The ratchet was re-recorded** (`tests/fusion_quality_baseline.json`), and the
diff is DTCWT's alone. One metric moved past its tolerance the wrong way:
`long_stack/seam_excess`, 11.06 -> 11.39 against a tolerance of 0.25. It is the
sharper picture rather than a lattice - the same fixture gained 2.82 dB of PSNR,
0.009 of SSIM and 0.52 of spatial frequency, and the two metrics that measure
steps in *flat* areas both improved there (`defocus_seam_excess` 13.39 -> 13.08,
`defocus_seam_visible` 20.2% -> 19.7%). `deep_stack` moves the other way by a
third of a tolerance on its two seam severities (+0.026 and +0.035) and improves
on speckle (2.03% -> 1.88%) and on visible seams (2.52% -> 1.43%); everything
else in the diff is a fraction of a tolerance.

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

**Status at 1.11.0: 25%.** The interim recommendation shipped: the render
pipeline now requests the GPU by default (`core/workers.py` constructs
`MultiFocusFusion(..., use_gpu=True)`, with automatic CPU fallback when torch
or a device is missing). The CPU scaling itself is unchanged, pending items 3
and 7.

**Both of those have landed now, and neither is the cause.** Item 3 took 24.5%
off the CPU time in 1.11.1 and item 7 a further 6-15% in 1.30.13, but the
growth itself did not move: 6 frames at 512, 1024 and 2048 px multiply by 5.5x
then 5.3x on the pairwise code and 5.7x then 5.4x on the joint selection, so
the superlinearity is in the transform and its memory traffic rather than in
the fusion this document has been changing. The `dtcwt` package's NumPy
transform is what is left to look at, and that is a different piece of work
from anything in this file. The item stays at 25%.

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

## 11. DCT fails outright on stacks of 256 frames or more

**Category: quality. Impact: high. Effort: low. Fixed in 1.11.2.**

`dct.py` stores its per-block winner map as `uint8` while the stack has fewer
than 256 frames, and widens it to `int32` beyond that:

```python
idx_dtype = np.uint8 if len(images) < 256 else np.int32
```

The wide path then fails twice downstream:

- The consistency filter converts the map to float32 and median-filters it
  with the default `kernel_size=7`. OpenCV allows apertures above 5 on 8-bit
  input only, so the call raises. Measured: fusing 260 random 64x64 frames at
  the default kernel fails with `cv2.error` from `median_blur.simd.hpp`
  (OpenCV 5.0.0) before producing anything.
- With `kernel_size` <= 5 the filter runs, but reconstruction casts the map
  with `final_index_map.astype(np.uint8)` "because resize is fastest on
  uint8", wrapping every index above 255 (index 300 becomes 44 - measured).
  Pixels whose winner wrapped are then filled from the wrong frame, or - when
  the wrapped value matches no surviving index - left black.

A stack this deep is not hypothetical: video import routinely produces
hundreds of frames.

**Fix.** Keep the map in `uint16` end to end. `cv2.medianBlur` accepts 16-bit
input at apertures 3 and 5 (iterate the 5-aperture filter when a larger kernel
is requested), and `cv2.resize` with `INTER_NEAREST` handles `uint16`
directly, so the downcast can simply be deleted. Whatever lands must be
mirrored in `dct_torch.py`, which shares the design.

**Fixed in 1.11.2**, as prescribed: the winner map is `uint16` from allocation
through filtering, resize and reconstruction, and the `astype(np.uint8)`
downcast is gone. Median filtering goes through a shared
`_median_filter_index_map` helper: apertures 3 and 5 run directly on the
16-bit map; larger kernels iterate the 5-aperture filter enough times to
cover the requested radius (two passes for the default kernel of 7). The
same helper and `uint16` map now drive the consistency step in
`dct_torch.py`, which previously took a lossy `float32` median for stacks of
256+ frames and no longer needs cv2 directly. Verified: 260 random 64x64
frames at the default kernel now fuse without error, indices above 255
survive the filter unchanged, and the DCT tests in
`tests/test_fusion_quality.py` and `tests/test_fusion_characteristics.py`
still pass.

---

## 12. DTCWT lets a pixel's colour channels come from different frames

**Category: quality. Impact: medium. Effort: low. Fixed in 1.14.0 and 1.30.14.**

The R, G and B channels are transformed and fused in three independent passes,
each with its own activity masks; nothing ties a pixel's channels to the same
source frame. At a depth edge the red channel can be taken from one frame and
blue from another, producing a colour that exists in no source - chromatic
fringing. Every other method here guards against exactly this: `depthmap.py`
and `pyramid.py` sum squared responses across channels first so "colour can
never split across sources", and item 1's fix kept a single decision map for
all channels of GFG-FGF.

**Fix.** Compute the activity and consistency masks once - on luminance, or as
the maximum across the three channels' coefficient magnitudes - and select all
three channels with the same mask. Mask construction drops from three passes
to one, and the change folds naturally into item 7's joint-selection rewrite.

**The detail bands have taken the second of those since 1.14.0**, in both
paths: the channels are reduced with a maximum before anything is compared, so
one decision selects all three (`_coefficient_activity` in `dtcwt.py`,
`dtcwt_torch.py`), and item 7's rewrite kept it. That is the half of the method
this item was written about, and it is pinned by
`test_dtcwt_frame_order.py::TestFold::test_all_channels_of_a_coefficient_follow_one_decision`.

### The coarse band had it back, and nobody had looked

Item 16 replaced the lowpass band's plain mean with an activity-weighted one in
1.11.3, and built **one weight map per colour channel** - `_lowpass_activity`
was called once per channel and the results stacked. So each channel mixed the
stack in its own proportions, which is this item's defect at large scale: the
detail bands could no longer split a pixel's colour, but the band underneath
them could. `pyramid.py` never had this - `_band_energy` sums the squared
responses across channels into one grey map, and its base weight is that map -
so DTCWT was the only method left doing it.

How far apart the channels' mixes actually were, over every fixture the project
has. "Channels disagree" is the share of coarse-band pixels where the three
channels' dominant frame is not the same frame; the distance is the largest
total-variation distance between two channels' mixes over the stack, where 0 is
one shared mix and 1 is no overlap at all:

| scenario | frames | channels disagree | distance mean | p99 | max |
|---|---|---|---|---|---|
| fine_texture | 3 | 0.12% | 0.073 | 0.190 | 0.228 |
| sensor_noise | 3 | 1.94% | 0.070 | 0.224 | 0.272 |
| depth_edge | 2 | 17.00% | 0.035 | 0.192 | 0.238 |
| long_stack | 12 | 2.19% | 0.076 | 0.287 | 0.338 |
| low_contrast | 3 | 1.38% | 0.048 | 0.165 | 0.198 |
| saturated_colour | 3 | 40.69% | 0.078 | 0.384 | 0.430 |
| deep_stack | 64 | 55.69% | 0.074 | 0.304 | 0.387 |
| photographic | 5 | 5.21% | 0.040 | 0.127 | 0.178 |

**What that is worth in colour.** Mixing two frames in one proportion traces the
straight line between their colours; mixing each channel in its own proportion
leaves that line, and the distance from it is colour the fusion invented. On
the two-frame fixtures, reconstructing the coarse band alone (the highpasses
zeroed, so nothing but the mix is in the picture) and measuring that distance
in levels:

| fixture | mean before | after | p99 before | after | max before | after | >1 level before | after |
|---|---|---|---|---|---|---|---|---|
| depth_edge | 0.073 | 0.009 | 0.889 | 0.104 | 1.206 | 0.337 | 0.41% | 0.00% |
| chromatic detail | 0.006 | 0.004 | 0.068 | 0.044 | 0.258 | 0.163 | 0.00% | 0.00% |
| + 22% exposure drift | **3.432** | **0.035** | 5.193 | 0.193 | 5.986 | 0.413 | **98.86%** | **0.00%** |

The third row is the fixture that actually exercises the defect, and it is item
16's own caveat: the synthetic scenarios hold brightness constant across frames,
so the frames barely disagree at coarse scale and there is little for a split
mix to get wrong. Darken one frame by 22% and 98.86% of the coarse band is more
than a level off any colour the two frames carry, by up to 6 levels. What is
left after the fix is float residue - a fifth of a level at the 99th percentile.

**Fixed in 1.30.14**, in both paths: `_lowpass_weight` sums the three channels'
aggregate activity into one map, and that map weights all three channels
(`dtcwt.py`, `dtcwt_torch.py`). The detail bands' rule is unchanged.

**Why a sum here when the detail bands use a maximum.** The detail bands
*select* - the question is which frame owns a coefficient, and a maximum keeps a
structure living in one channel from being diluted by two flat ones (item 1's
lesson). The coarse band *weighs*, and there the sum is the aggregate the
question actually asks for; it is also what `pyramid.py` and `depthmap.py`
already use. Both were measured, along with a luminance weighting. They are
level to two decimal places on eight of the nine fixtures; `deep_stack`
separates them, and the maximum is the worst of the three:

| reduction | deep_stack PSNR | colour error |
|---|---|---|
| maximum, per level | 25.80 | 7.85 |
| maximum, of the totals | 25.88 | 7.78 |
| **sum** | **25.94** | **7.74** |
| luminance | 26.01 | 7.69 |

Luminance edges it, and is rejected anyway: weighting blue at 0.114 is the
dilution item 1 exists to prevent, and it would have this method answering a
question about *detail* with a measure of *brightness*. The sum also commutes
with the area resampling that follows it, so unlike the maximum it cannot
matter whether the channels are reduced before or after the resize - the two
readings of the same sentence are the same computation.

**What it costs.** PSNR against each fixture's own reference, CPU path:

| scenario | before | after | delta |
|---|---|---|---|
| fine_texture | 38.79 | 38.79 | 0.00 |
| sensor_noise | 45.14 | 45.16 | +0.02 |
| depth_edge | 37.31 | 37.32 | +0.01 |
| long_stack | 35.46 | 35.47 | +0.01 |
| low_contrast | 65.00 | 65.00 | 0.00 |
| saturated_colour | 33.65 | 33.67 | +0.02 |
| deep_stack | 26.02 | **25.94** | **-0.08** |
| photographic | 37.66 | 37.66 | 0.00 |

`deep_stack` is the only fixture that pays, and it is the one where the coarse
band carries the most: 64 frames, a background no frame resolves, and a
reference that declares the least-blurred rendering correct. 0.08 dB is a ninth
of the ratchet's tolerance. Everything else is level or fractionally better, and
the colour error over the flattest half of each frame - where the coarse band is
what is being looked at - moves by less than 0.01 everywhere except `deep_stack`
(+0.072) and the exposure-drift fixture (**-0.490**). The quality ratchet needed
no re-recording: one metric of the 56 moved by more than a fifth of its
tolerance, `deep_stack/block_speckle`, and it improved (1.875 -> 1.719, which is
one 8x8 cell). Runtime is unchanged - 0.996 to 1.017x over four stacks from 6
frames at 768x768 to 24 at 512x512, which is noise.

**Guarded by** `tests/test_dtcwt_colour_channels.py`. The unit tests pin
`_lowpass_weight` - one map rather than three, blind to *which* channel carries
a structure but not to how many do. The render test builds a stack whose middle
is a smooth field with no detail in it at all, so what is rendered there is the
coarse band alone, and asserts the fused colour lies on the line between the two
frames' own colours: 0.40 levels off on average and 1.09 at worst, against 2.76
and 8.09 on the per-channel weight. All seven fail on the 1.30.13 code.

**One pixel of `saturated_colour` is the price, and it is a quantiser, not a
rule.** That fixture used to render byte for byte however the stack was ordered,
and now one pixel of its 102,400 moves by one level. The cause is not this
change: 571 of its coefficients are won on a tie in *both* the regional activity
and the coefficient magnitude, by frames that differ only in phase - item 7's
documented residual, which it declines to break with an arbitrary rule about
complex numbers - and those ties already made the float picture depend on the
order by 1.7e-7. Moving the coarse band by that much put one pixel on exactly
21.5. The GPU path is still byte-exact there.
`tests/test_dtcwt_frame_order.py` now asserts what the fixture can carry: a
reordering may round at most a handful of pixels by at most one level, and
nothing else may move.

---

## 13. The folder loader is copy-pasted four ways, and two inputs crash it

**Category: robustness. Impact: low. Effort: low.**

`dtcwt.py`, `pyramid.py` and `depthmap.py` each carry the same private folder
loader; `dct.py` has a fourth with different semantics. The shared copy has
two defects, both measured:

- Its sort key returns `int` for numbered names and `str` otherwise, so a
  folder holding both (`img1.png` beside `cover.png`) raises
  `TypeError: '<' not supported between instances of 'str' and 'int'`.
- It infers a single extension from the first directory entry and globs only
  that, so a mixed-extension folder silently loses every other format - while
  `dct.py`'s variant accepts six extensions. Same app, two behaviours.

The GUI is unaffected - it always hands the methods decoded arrays via
`ImageStackLoader` - which is why this has survived. It bites direct API
callers only.

**Fix.** One shared helper in `utils/`, with a type-stable sort key
(`(0, number)` / `(1, name)` tuples) and the same extension handling
everywhere.

**Half of that already exists.** Registration had the same defect - three
copied loaders, drifted extension lists, a sort key that returned a bare `int`
or `str` - and [REGISTRATION_IMPROVEMENTS.md](REGISTRATION_IMPROVEMENTS.md)
item 12 fixed it in 1.30.10 by moving the loader into `utils/image_utils.py`
rather than writing a fourth private one: `supported_input_extensions`,
`stack_sort_key` and `load_image_folder`, the extension set derived from what
`read_image_any_depth` can decode. This item is now adopting those three in
`dtcwt.py`, `pyramid.py`, `depthmap.py` and `dct.py`, not designing them.

---

## 14. DTCWT's consistency vote is biased at image borders

**Category: quality. Impact: low. Effort: trivial. Dissolved in 1.30.13.**

The majority filter counts each pixel's agreeing neighbours with
`convolve(..., mode='constant', cval=0.0)` and compares against a fixed
`window^2 / 2` threshold. Border pixels have fewer real neighbours but face
the same threshold, so votes for the first source are systematically
undercounted there, and the border strip leans toward the second source
regardless of sharpness.

**Fix.** Reflect the neighbour count at the border, or threshold against the
true neighbour count. Note the interaction with item 3: its `cv2.boxFilter`
substitution landed in 1.11.1 with parity kept deliberately
(`borderType=BORDER_CONSTANT`), so this bias survives that swap unchanged
and remains open to fix on its own terms.

**Dissolved in 1.30.13, rather than fixed.** Item 7 replaced the vote with a
comparison of pooled activities, so there is no count and no fixed threshold
left to be biased - both remaining filters treat the frame edge identically for
every frame. Measured on pairs of frames of the same statistics, where half
each is the honest answer, the first frame's share of the border ring goes from
27.0-29.0% to 49.6-52.5% against an interior of 49.2-50.5%. The measurement is
in item 7; nothing was written for this item on its own.

---

## 15. Depth-map MODE_MAX ships its decision map unregularised

**Category: quality. Impact: low. Effort: low.**

DCT median-filters its index map twice and DTCWT majority-votes its masks, but
the depth map's hard per-pixel select relies on box-filter pooling alone.
Where two frames' pooled energies are nearly equal - smooth transitions
between depth planes - the winner can alternate pixel to pixel and speckle the
seam. The tools this method cites (Zerene DMap, Helicon A/B) all smooth their
depth maps before gathering.

**Fix.** An optional median (or guided-filter) pass over the winning-index map
before the gather, matching the regularisation the other selectors already
have. MODE_AVERAGE needs nothing: blending is its own smoothing.

---

## 16. DTCWT and Pyramid average the lowpass band across all frames

**Category: quality. Impact: low. Effort: medium. Fixed in 1.11.3.**

Both methods collapse the coarse residual by a plain mean over the stack
(`np.mean(lowpass_stack)` in one, `base_accumulator / num_images` in the
other). Focus stacks mostly share their low frequencies, so this is usually
harmless - but when frames disagree at large scale (exposure drift, strong
focus breathing) the mean ghosts that disagreement into the result. The
standard remedy is to weight each frame's base by its aggregate detail
activity, so the sharpest frames dominate the coarse band too.

Measure on a real stack first: the synthetic scenarios hold brightness
constant across frames, so they cannot show this failure - the same caveat
item 6 carried.

**Fixed in 1.11.3**, as prescribed: the coarse band is now a per-pixel
weighted average, each frame weighted by its aggregate detail activity
resampled to the coarse grid, plus an epsilon (1e-6) so a stack with no
detail anywhere degrades to the old plain mean. In `pyramid.py` the weight is
the running sum of the band energies already computed for choose-max, cascaded
down by `pyrDown` to the base grid; in `dtcwt.py` (and its GPU twin
`dtcwt_torch.py`, where the weighted sum streams frame by frame like the plain
sum did) it is the highpass magnitudes summed over the six orientations and
all levels, area-resampled to the lowpass grid.

**One detail of this was wrong until 1.30.14.** DTCWT built the weight once per
colour channel, so each channel mixed the stack in its own proportions - item
12's defect, reintroduced in the band item 12 had not been looked at in. The
channels are now summed into one weight map before it is applied, which is
what `pyramid.py` had been doing here all along; the measurement is in item 12.

Measured on the caveat's own failure case - a two-frame stack, each half sharp
in a different frame, the second frame 20% darker: mean absolute error against
a reference that keeps each half at its sharp frame's brightness fell from
13.0 to 1.4-2.7 (Pyramid) and from 12.9 to 3.9-5.5 (DTCWT). On a
constant-brightness stack the output is unchanged for practical purposes (old
vs new agree at 87 dB / 59 dB PSNR), the full quality suite passes, and
`test_gpu_matches_cpu[dtcwt]` confirms the CPU and GPU paths still land
together.

---

## 17. DCT scores contrast rather than sharpness, so defocused washes win

**Category: quality. Impact: high. Effort: low. Fixed in 1.15.1 and 1.15.2.**

Reported from a macro render: next to the pyramid result, DCT was visibly
sharper but carried large flat patches of a single tone where the pyramid had
background detail. The patches sat on the block grid and had stepped edges, so
they were selection failures, not a filtering artefact.

Two independent causes turned out to produce the same artefact, and the second
only became visible once the first was fixed. Both are below, in the order they
were found.

The measure was the cause. `dct.py` scored a block by its plain pixel variance,
which by Parseval is the energy of *all* its AC coefficients:

```python
mean_sq = cv2.resize(gray ** 2, (map_w, map_h), interpolation=cv2.INTER_AREA)
mean_val = cv2.resize(gray, (map_w, map_h), interpolation=cv2.INTER_AREA)
var_map = mean_sq - mean_val ** 2
```

That answers "how much contrast is in this block", which is not the same
question as "how sharp is it". A heavily defocused highlight spreads a smooth
brightness ramp across the frame, and a ramp crossing one block carries more
energy than the fine, low-amplitude texture that is genuinely in focus there -
so the *blurred* frame wins, and its pixels are copied through verbatim. The
consistency filter cannot undo it: the regions are far larger than a median
kernel, so it consolidates them into clean-edged blobs instead of removing them.
This is also why the paper's own consistency step exists at all, and why the
method has always been described here as the blockiest of the six.

Measured on a fixture built from the reported case - dark background carrying
faint fine markings, a bright object at another depth whose defocus washes over
it:

| | PSNR | background taken from the wrong frame |
|---|---|---|
| Best single unfused frame | 26.05 dB | - |
| **DCT, before** | **26.60 dB** | **35.7%** |
| Pyramid | 29.85 dB | 20.6% |
| **DCT, after** | **29.35 dB** | **2.0%** |

**Fix, part one (1.15.1).** Band-limit the measure from below and pool it before
choosing, which is what `pyramid.py` already does (`_band_energy`) and why it
does not show this failure:

- **High-pass first.** Subtract a box blur the width of one block, then take the
  block mean of the squared detail. Structure coarser than a block - exactly the
  low AC band a defocused wash lives in - is gone before the energy is measured,
  so a wash scores near zero however bright it is. The detail image is zero-mean
  by construction, so the `E[X]^2` term is no longer computed at all.
- **Pool over 3x3 blocks** before the choose-max, so the decision is regional
  rather than per-block and grain cannot flip it.

**What part one missed.** Re-rendered on the reporter's stack, the large patches
were gone but small flat ones remained - and they all shared a single cream
tone, which is the tell: they were coming from one frame, not from many wrong
decisions. That frame was a near-uniform bright veil, one of the frames focused
far in front of the subject, holding no detail anywhere.

It was winning on grain. Photon noise variance grows with signal, so in a block
where nothing is in focus a bright frame carries more high-pass energy than a
dark one purely as noise. Measured on flat 8-bit patches with a realistic sensor
model, energy ratios of bright to dark:

| dark level | veil level | high-pass energy ratio |
|---|---|---|
| 30 | 232 | 7.8x |
| 55 | 210 | 3.8x |
| 20 | 245 | 12.2x |

Nothing in part one catches that. The veil wins by 4-12x, which reads as a
decisive result rather than a tie, and its energy is ten to a hundred times the
absolute noise floor the code was testing against. The regional fallback made it
worse: it decided undecidable blocks by the same energies at a wider scale, so
it re-ran the same noise contest and merely smoothed the outline of the answer.

**Fix, part two (1.15.2).** Make the comparison fair, then stop asking the
energies once they have nothing to say:

- **Score each frame against its own noise.** Divide a frame's block energies by
  the 10th percentile of those energies - the level of its own quietest region,
  which in a focus stack is always somewhere out of focus. The measure becomes a
  signal-to-noise ratio, about 1.0 wherever a frame resolves nothing, whatever
  its brightness. The veil's advantage disappears, and blocks that hold no
  detail tie instead of being won.
- **Treat "no detail" as its own outcome.** A block is undecided when the winner
  is under 2x its own noise, or when it fails to beat the runner-up by 10%. On
  the fixtures the winner's SNR runs 1.1-2.2 across genuinely featureless
  regions and never drops below 93 in the detailed scenarios, so the threshold
  sits in a wide empty gap rather than on a slope.
- **Propagate rather than re-decide.** Undecided blocks take the choice of the
  nearest decided block (`cv2.distanceTransformWithLabels`, seed labels read
  back out of the transform's own output so nothing depends on OpenCV's
  numbering). Where no frame resolves anything there is no focus information to
  recover, and continuity with the surroundings is the only defensible answer.
  The 9-block fallback that part one added is gone.

Both mechanisms are needed; neither works alone. Ablated on the veil fixture,
percentage of the smooth dark body covered by veil patches:

| | veil patches | PSNR |
|---|---|---|
| Neither | 97.7% | 11.67 dB |
| Normalisation only | 80.4% | 14.57 dB |
| Propagation only | 48.7% | 6.76 dB |
| **Both** | **0.0%** | **31.30 dB** |

The whole change is O(1) in stack depth: only the running top two energies and
one index map are held, never a per-frame stack.

**Effect on the rest of the suite.** Better on five of the six scenarios and
level on the sixth: low_contrast 51.61 -> 60.33 dB, depth_edge 31.93 -> 35.40,
saturated_colour 24.81 -> 30.03, long_stack 27.62 -> 27.83, fine_texture 31.64
unchanged. sensor_noise costs 0.45 dB (38.62 -> 38.17) - a high-pass measure
sees grain more clearly than a variance measure does, so the method stays the
weakest of the classical three on a noisy stack, as
`test_dct_struggles_most_with_noise` still asserts.

**Cost.** 217 -> 306 ms for 8 frames of 1600x2400 on CPU (+41%): one extra
full-resolution box blur and a percentile per frame, against one fewer block
resize. DCT remains by far the fastest method - the same stack takes 852 ms
through the guided filter, 965 ms through the pyramid and 1031 ms through
GFG-FGF.

**Side effects on two other items**, both measured rather than assumed:

- **Item 4 improves but is not settled.** Agreement across four shuffles of a
  12-frame stack rises from 30.2 to 33.6 dB: exact ties no longer fall to the
  lowest index, and propagated blocks are decided by position rather than by
  which frame arrived first. What remains order-dependent is the near-tie that
  clears the 10% margin, so the item stays open.

  **Overtaken by item 18, then closed by item 4 itself (1.30.12).** The margin
  rule this describes was removed a version later, and what replaced it made the
  order load-bearing rather than incidental - agreement fell back to 25.1 dB on
  `deep_stack` before the run rule took it to 34.3. See item 4 above for where
  that ended up and why full invariance is not the goal any more.
- **`kernel_size` no longer does anything on these fixtures.** The decision map
  now reaches the median filter already regionally coherent, so the consistency
  step has nothing left to remove: across kernels 3 to 31 the result is
  identical on `long_stack` (the old code varied by 0.61 dB there),
  `fine_texture` and `sensor_noise`. The slider has arrived in item 9's
  territory - a dial that no longer changes anything - and should either be
  hidden for this method or given something to do.

  **Overtaken by item 18 (1.17.1).** The rewrite made this the slider that
  regularises the focal-plane map, and it is now the method's main quality
  control rather than a spare part - the paragraph above describes code that no
  longer exists. Measured on a 274-frame stack, the share of flat-area blocks
  standing out from their neighbours runs 0.385% unfiltered, 0.333% at 7,
  0.304% at 31 and 0.281% at 51. See "Tuning DCT" below.

Both paths landed together, as item 11 requires. `dct_torch.py` reproduces the
box blur with a reflect-padded `avg_pool2d`, and the decision stage - normalise,
gate, propagate, median-filter - runs on the host through the CPU path's own
helpers, on a map of a few thousand entries, so the two cannot drift apart
there. `test_gpu_matches_cpu[dct]` passes; CPU and GPU are bit-identical on the
quality fixture and on the wash fixture, agree at 41.5 dB on the veil fixture
(`torch.quantile` and `np.percentile` estimate the noise level slightly
differently, which flips a handful of near-tied blocks), and agreement on 260
random frames - an input made entirely of near-ties - improves from 7.8 to
16.0 dB.

**Guarded by** `tests/test_dct_defocus_wash.py`, which carries both fixtures and
fails on the pre-1.15.1 code (35.7% and 99.9% of the respective regions taken
from the wrong frame) and on the 1.15.1 code for the veil fixture (37.2%).

---

## 18. DCT's per-block winner stops existing in a deep stack

**Category: quality. Impact: high. Effort: medium. Fixed in 1.17.1.**

Reported against a 274-frame macro stack (2048x1364, no tiling - the longest
side has to *exceed* `tile_threshold`, and 2048 does not). Item 17's patches
were gone; in their place the defocused background was torn into chunks with
stepped edges, while the pyramid rendered the same background smoothly.

Instrumenting the decision map explained it in one line: **91% of blocks were
undecided**. Item 17 settled a block by requiring the winner to beat the
runner-up by 10%, and in a stack this deep no frame ever does - neighbouring
frames are one focus step apart and differ by far less than that. So the margin
test abstained almost everywhere, and what actually drew the picture was the
fallback: give the block its nearest decided neighbour's frame.

That fallback is fine when it fires on a few blocks and catastrophic when it
fires on nine in ten. The nearest decided block can be anywhere in the stack -
measured neighbour-to-neighbour jumps ran to **271 frames**, with 6.6% of block
borders jumping more than 10 - and two frames that far apart look nothing alike
in a defocused area. Every such jump is a visible step, on the block lattice.

The lesson generalises past this bug: **a rule that abstains has to be judged on
what happens when it abstains**, and the synthetic scenarios could not have
shown this because the deepest was 12 frames.

**Fix.** Stop asking which single frame wins and ask where the focal plane is.

- **Plateau, not argmax.** Every frame within `_PLATEAU` of the peak energy is
  in focus on that block - its depth of field - and the middle of that set is
  the focal plane, to sub-frame accuracy. In focus the set is a short run and
  the middle is the true plane. Out of focus every frame is equally poor, the
  set is the whole stack, and the middle is mid-stack: the same answer
  everywhere, so a defocused region comes out uniform instead of patchwork.
  There is no abstention and so no fallback to misbehave.
- **Composite the weights, not the labels.** Each frame is weighted by how close
  the map is to it, and the *weights* are upsampled from the block lattice to
  pixels. Upsampling an index can only step from one frame to the next at a
  block edge; upsampling a weight ramps between them across the block. Where a
  region shares one frame the weight is exactly 1 and the pixels come through
  untouched.
- **Keep the median.** Without it a speck of dust that is sharp in exactly one
  frame drags its block onto that frame, which shows up as a bright square in an
  otherwise defocused area - seen, and fixed, during this work.

Measured on the reported stack, against the pyramid the report was compared to.
"Background step" is the mean step on the block lattice in defocused areas
beyond what the local content justifies, in 8-bit levels
(`fusion_metrics.defocus_seams`):

| | background step | lattice visible | detail |
|---|---|---|---|
| Pyramid | 0.456 | 1.65% | 18.03 |
| **DCT, before** | **0.996** | **4.80%** | **15.21** |
| **DCT, after** | **0.286** | **1.45%** | **16.04** |

The tearing is gone, and on its own complaint the result now beats the pyramid.

**The trade, stated plainly.** A region that is never in focus in *any* frame -
a background beyond the stack's reach - is now rendered from the middle of the
stack rather than from whichever frame happens to render it least blurred. It is
smoother and slightly flatter. On the `deep_stack` scenario, whose reference
declares the least-blurred rendering to be correct, that costs **2 dB** against
the old code (34.5 -> 32.6 at plateau 0.7). Widening the plateau buys most of it
back - 0.80 scores 34.8 - but past 0.85 the veil artefact of item 17 returns
(22.5% of the fixture at 0.90), so `_PLATEAU` sits at 0.80 with margin on the
side that produces an artefact rather than a softness.

**Cost.** Two measurement passes instead of one, plus a compositing pass:
274 frames of 2048x1364 take 14.4 s against 5.3 s, with the pyramid at 18.5 s on
the same stack. DCT is no longer the fastest method by a wide margin, only a
modest one.

**Output is no longer verbatim.** DCT used to copy each winning block's pixels
through untouched. It now blends between neighbouring frames, which is what
removes the last of the staircase. The frames blended are one focus step apart,
so no detail is lost to it, but the "every output pixel is exactly some input
pixel" property is gone and `test_dct_defocus_wash.py` now asserts the weaker
contract that holds: every pixel stays inside the envelope its sources span.

**Guarded by** `tests/test_fusion_regression.py` - see below. Both halves of
this fix trip it when removed.

**Refined by item 4 (1.30.12)** in two places this section describes. "The
middle of that set" is now the middle of the *run* containing the peak, because
a set gathered from the whole stack can include a frame that clears the bar on
grain a hundred frames away. And "to sub-frame accuracy" was true of the
estimate but not of the output: the map was rounded to a whole frame before the
median filter, and is now carried in halves the rest of the way.

---

## 19. The pyramid's choose-max has no answer where nothing is in focus

**Category: quality. Impact: high. Effort: medium. Fixed in 1.19.0.**

Reported as thin dark strokes over the smooth, defocused top-right corner of a
macro render - filaments crossing a background that no frame resolves and no
frame contains.

The published rule is choose-max: for every band of the Laplacian
decomposition, copy the coefficient with the most local energy and discard the
rest. Where one frame is plainly sharper that is the right answer. Where none
is - which on a macro frame is most of the picture - the energies differ only
by grain, and three separate things go wrong at once:

- **The winner map becomes a speckle field.** Grain decides it, so neighbouring
  pixels take their coefficients from frames that disagree about what is there.
- **The bands of one pixel disagree with each other.** Each band chooses
  independently, so a pixel can take its fine detail from frame 3 and its
  coarse structure from frame 40.
- **The sum of those choices is not near any frame.** Collapsing the pyramid
  adds bands from different frames, and nothing in the sum keeps the result
  inside the range the stack spans. On `deep_stack` the render came back up to
  **24 levels darker than the darkest source**, with 2.05% of the frame more
  than 8 levels under. A pixel darker than every frame is, by definition,
  something the reconstruction invented - and over a smooth background a
  connected run of them is exactly the reported stroke.

A fourth problem was already known from DCT: photon noise grows with
brightness, so a bright defocused veil carries several times the band energy of
a dark sharp frame *in grain alone*, wins every detail-free region and stamps
its flat tone across them. That is item 17, in a method nobody had checked for
it. On the veil fixture from `tests/test_dct_defocus_wash.py` the pyramid gave
**32.3% of the subject's smooth body** to a veil frame, and scored 18.5 dB.

**Fix.** Four changes, each aimed at one of the above.

- **Weight instead of choose.** Each frame contributes to a band with weight
  `(its energy / the best energy) ** selectivity`. At the default of 8 a frame
  2x behind the winner contributes 0.4%, so a real focus decision is still a
  decision; where frames tie they average, which is the correct answer for
  "nothing is in focus here" and takes the speckle field with it. `inf`
  restores the published rule exactly.
- **Compare in units of the frame's own grain.** Each band is divided by a low
  percentile of its own energies, so every frame sits at about 1.0 where it
  resolves nothing and the veil cannot buy regions with grain. Pixels at
  exactly zero are left out of that percentile: a clipped-black backdrop is
  zero over a large part of every frame, and reading the level off those pixels
  returns zero however much grain the rest carries.
- **Clamp to the source envelope.** Every pixel is held between the darkest and
  brightest value its own frames have there. The true all-in-focus value comes
  from whichever frame resolves the pixel, so it is one of the sources by
  construction and any blend of them lies between - clamping removes only
  values no frame supports. It costs a running min and max over the stack, two
  frames of memory. The block methods get this for free by compositing source
  pixels; a pyramid composites coefficients, so it has to be imposed.
- **Let the detail carry the base.** Item 16 replaced the base band's plain
  mean with a weighting by aggregate activity, which is still far too gentle
  when most of the stack is defocused: on the veil fixture 24 frames of haze
  outvote the few that resolve the subject. Raising the exponent to 3 fixes
  that (18.5 dB at the plain mean, 30.4 at the old weighting, 35.1 at 3, flat
  past 4).

Measured on every fixture the project has. "Under" is the worst pixel below the
darkest source, in 8-bit levels, and the share of the frame more than 8 levels
under it:

| scenario | PSNR before | PSNR after | under before | under after |
|---|---|---|---|---|
| fine_texture | 45.45 | 46.05 | 15 / 0.04% | **0** |
| sensor_noise | 51.15 | 51.94 | 9 / 0.00% | **0** |
| depth_edge | 35.77 | 37.86 | 5 / 0.00% | **0** |
| long_stack | 45.68 | 46.12 | 20 / 0.03% | **0** |
| low_contrast | 68.05 | 69.44 | 1 / 0.00% | **0** |
| saturated_colour | 33.12 | 33.89 | 28 / 0.10% | **0** |
| deep_stack | 32.85 | **37.31** | 24 / 2.05% | **0** |
| veil | 18.46 | **35.12** | 2 / 0.00% | **0** |

Every scenario improves, the two that hold the artefacts by 4.5 and 16.7 dB,
and the invented pixels are gone everywhere. The veil frames take 0.0% of the
smooth body, against 32.3% before.

**Cost.** A weighted sum is more arithmetic than a masked copy: 24 frames of
2048x1364 take 3.6 s against 1.6 s, and `selectivity=inf` runs the old path at
the old speed (1.4 s). The weights are computed in one streaming pass - the
running peak is tracked as frames arrive and the accumulated sums are rescaled
whenever it rises, which is exact rather than approximate - so memory is still
bounded by the accumulators plus one frame's pyramid, and the stack depth does
not enter into it.

**One metric moved the wrong way.** `defocus_seam_visible` on `deep_stack` rose
from 1.5% to 2.7%. It counts steps exceeding three times the *local* typical
step, and the background is now smooth enough that the local step it is
measured against halved: `defocus_seam_excess`, the severity, fell from 0.466
to 0.257. `block_speckle` rose by 0.156, which is one 8x8 cell out of 640 - the
metric's own quantum, and finer than the tolerance the ratchet gives it. The
reference image scores 2.188 on that metric itself.

**Guarded by** `tests/test_pyramid_flat_field.py`, which fails on the pre-1.19.0
settings for both artefacts, and by the quality ratchet.

---

## 20. Depth Map runs on the CPU only, and sizes its pool by threads alone

**Method:** Depth Map | **Category:** Performance | **Impact:** Medium |
**Effort:** Medium | **Fixed in 1.31.0**

Every other classical method here grew a device path (items 3 and 8, and the
1.5.x-1.5.7 work behind Guided Filter, DCT and GFG-FGF). Depth Map did not, and
`MultiFocusFusion._validate_environment` said so in a comment while forcing
`use_gpu = False`. The method is the *cheapest* of the six per frame - one
3x3 Laplacian, one box filter, one comparison - which is exactly what makes it
memory-bound rather than compute-bound, and exactly the shape a GPU eats.

The pool that measures the frames had a second problem. Its size came from
`min(8, cpu_count)` or from whatever thread count the caller passed, and
nothing else: at 60 MP a single frame is 720 MB once it is float32, so eight in
flight reserve 5.7 GB that no part of the render had budgeted for. The sliding
window in `_map_in_order` bounds the count of live buffers, but the *size* of
each one is set by the image and was never consulted.

**What the measurements said.** 12 frames of 4000x3000, kernel 9, on the
machine described at the top:

| | Time | Peak RSS |
|---|---|---|
| Max, before | 1.11 s | +3709 MB |
| Max, after | 1.08 s | **+2624 MB** |
| Average, before | 2.35 s | +3577 MB |
| Average, after | **1.99 s** | **+2844 MB** |

Nearly all of the 1085 MB is one change: MODE_MAX was accumulating the winning
*pixels* in a float32 BGR buffer (12 bytes per pixel) and each measurement task
was holding its normalised frame alive until the reduction consumed it. It now
accumulates the winning frame *index* (one byte per pixel), and gathers the
result from the sources afterwards at the stack's own depth. The output is
bit-identical - the float round trip it replaces was already exact - which the
quality ratchet confirms and a direct array comparison against the previous
implementation verified across both modes, both depths, three kernel sizes, two
halo radii, grayscale and mixed-depth stacks. MODE_AVERAGE gained its 0.36 s
and 733 MB from folding the weighting
and the final blend in place rather than through freshly allocated full frames.

The pool is now capped by `WORKER_MEMORY_SHARE` (25%) of the memory psutil
reports free, from an estimate of what one in-flight task actually holds. It
only ever lowers the count, and says so once when it does.

**The device path** (`fusion_methods/depthmap_torch.py`) mirrors the CPU one:

| | CPU | GPU | |
|---|---|---|---|
| Max | 1.12 s | 0.14 s | **7.9x** |
| Average | 2.03 s | 0.16 s | **12.8x** |
| Max, halo_radius 8 | 1.34 s | 0.21 s | **6.4x** |

Three things were worth care rather than speed:

*The chunk size is derived, not pinned.* `gff_torch` processes four frames per
batch because four was a number that fit. Here the batch is sized from
`cuda.mem_get_info` against an estimate of what a frame costs on the device -
15 frames for a 12 MP stack on a 16 GB card, 6 for a 24 MP stack with halo
suppression on, 16 (the cap) for the 1024 px tiles that tiled fusion feeds it -
and halves itself and retries on `OutOfMemoryError`, so being wrong is slow
rather than fatal.

*The tie-break survives batching.* The measurement is batched, but the
reduction still walks the chunk one frame at a time, because the strict `>` in
index order is what makes both paths pick the same frame where two tie. A
batched `argmax` would resolve those the other way, and items 4, 7 and 12 are
what order-dependence in a fusion method costs.

*The halo element is reproduced, not approximated.* `cv2.dilate` with
`MORPH_ELLIPSE` has no torch equivalent, and `max_pool2d` gives a square window
- which would push the halo band out to `r * sqrt(2)` at the corners and make
the radius mean something different on each path. The ellipse is decomposed
into one horizontal maximum per distinct row half-width (r+1 pooling passes for
2r+1 rows), reproducing OpenCV's `round(sqrt(r*r - dy*dy))` spans exactly;
`tests/test_depthmap_gpu.py` asserts bitwise equality with `cv2.dilate` for
radii 1, 2, 4 and 8.

Two departures from the CPU path remain, both at the frame border and both
deliberate: cv2 pools the energy with `BORDER_REFLECT` where torch's reflect
padding is `BORDER_REFLECT_101`, and MODE_AVERAGE sums a chunk's contributions
as a batch rather than one frame at a time. Inside the frame the two agree to
better than 55 dB.

**Guarded by** `tests/test_depthmap_gpu.py` - which runs the whole device path
on a `cpu` torch device, so the batching, the chunk loop and the dilation are
exercised without a card - and by the CPU/GPU parity pairs in
`tests/test_fusion_quality.py`.

---

## 21. Depth Map's pooling leaks across occlusion boundaries, ringing every subject

**Method:** Depth Map | **Category:** Quality | **Impact:** High |
**Effort:** Low | **Fixed in 1.33.0**

`_focus_energy` pooled the squared Laplacian with a single `cv2.boxFilter`.
A box window is edge-blind, so the enormous energy of a sharply focused,
high-contrast contour was smeared `kernel // 2` pixels in *every* direction -
including out across the occlusion boundary, onto background belonging to a
different slice. Inside that band the foreground's frame won the argmax, and
the result took the background from a frame where the background is defocused.
The visible artefact is a flat, washed-out ring hugging every subject outline,
its width set by the kernel dial: 25 px at kernel 51.

It went unnoticed because the method's default kernel is 9, where the band is
4 px, and because the ring is a *loss* of detail rather than something added -
it reads as "the background was never sharp there" unless another method is
put beside it. On a 333-frame macro stack of an ant on a PCB at kernel 51, the
silkscreen text ran right up to the ant in DTCWT, DCT and Pyramid, and
dissolved into a smooth blur roughly 25 px out from every leg and antenna in
Depth Map.

`halo_radius` made it worse rather than better, and by design: it grey-dilates
the *already pooled* energy, so the claimed band is `kernel // 2 + radius`
wide. The report that started this was a `k51_h30` render - a 55 px band.

**Fix.** Pool the same energy at two scales and take the geometric mean:

```python
wide = _pool(energy, window)                 # the dial's region measure
near = _pool(energy, window // 3)            # local evidence; barely leaks
energy = np.sqrt(wide * near)
```

The ratio between the two says how much of a frame's regional score really
belongs to this pixel's own neighbourhood - about 1 where the detail is
genuinely there, far below 1 in the band beside a contour the wide window has
reached across. Requiring a frame to carry both is what collapses the ring.
Where the energy really is uniform over the window the two poolings agree and
the mean is the plain pooled energy, so flat regions decide exactly as they
did. The narrow window is a *divisor* of the kernel rather than a fixed size,
so the correction stays the same fraction of whatever the caller dials in, and
it is dropped below 5 px: a window that small barely leaks, while a 3 px
pooling of a squared Laplacian is mostly sensor noise.

**A prefilter came with it.** A bare `ksize=3` Laplacian is the most
noise-sensitive high-pass there is, and on flat, low-signal regions the argmax
was ranking frames by their noise floor rather than their focus - which picks
whichever frame is most veiled, and mottles. Smoothing each frame by
`sigma=0.6` *for measurement only* (the output pixels are still gathered from
the untouched frames, so it costs nothing in output resolution) removes most of
that. 0.6 is where the ground-truth scenarios settle: it carries almost all of
the gain on `deep_stack` while a wider one starts costing on `sensor_noise` and
`fine_texture`.

**What the measurements said.** Every scenario in `tests/fusion_scenarios.py`,
PSNR against the ground-truth all-in-focus reference, MODE_MAX at kernel 51:

| Scenario | Before | After |
|---|---|---|
| fine_texture | 34.84 | **39.71** |
| sensor_noise | 47.78 | **50.32** |
| depth_edge | 34.39 | **35.45** |
| long_stack | 30.77 | **38.13** |
| low_contrast | 60.28 | **64.07** |
| saturated_colour | 30.40 | **31.65** |
| deep_stack | 27.60 | **29.49** |

No scenario got worse. On the ant stack, detail retained in the band around the
subject - measured against the per-pixel best any frame offers - improved 2.6x
(a deficit of 0.236 down to 0.092) with flat-region noise essentially unmoved.

**One metric moved the wrong way, and should have.** `deep_stack`'s
`spatial_frequency` fell from 17.35 to 14.48 while its PSNR rose 3.45 dB and
its SSIM rose 0.056. That scenario's background is never sharp in any frame,
and the old result filled it with the mismatched patches the scenario was
written to catch; the "detail" the metric was counting was the tearing. Put
beside the reference, the new background is the one that matches it.
`block_speckle` rose by 0.156 on two scenarios, which is one 8x8 cell out of
640 - the metric's own quantum.

**Cost.** One extra box filter and a separable 7-tap Gaussian per frame. The
333-frame ant stack at 1619x1064, kernel 51, 8 threads, best of three: 4.7 s
against 3.8 s.

**Guarded by** the quality ratchet (re-recorded, diff above),
`tests/test_depthmap_halo.py`, and the CPU/GPU parity pairs - MODE_MAX stays
bitwise identical between the two paths, MODE_AVERAGE within 2 LSB (the
pre-existing float summation-order departure, measured over 12 configurations).

---

## 22. Depth Map's argmax has no spatial prior, so it tears whatever it cannot resolve

**Method:** Depth Map | **Category:** Quality | **Impact:** High |
**Effort:** Low | **Fixed in 1.34.0**

Item 21 fixed what the focus measure claimed. This is about what happens where
it has nothing to claim at all.

Over a region no frame ever resolves - a background well behind the focus
sweep, or any dark, flat patch - every frame's energy sits at the same
defocused level and the winner is decided by whatever noise survived the
pooling. The argmax carries no spatial prior whatsoever, so neighbouring pixels
happily take frames from opposite ends of the stack. Those frames render that
background at visibly different blur and brightness, and the result is a torn
mosaic of hard-edged patches. It is the artefact `deep_stack` was written to
catch, quoting its own docstring: *"a tie broken badly shows up as steps
between frames that look nothing alike"*.

Measured on the reference ant stack, in a background region that never comes
into focus: **22% of pixels at kernel 9 chose a frame more than 3 slices from
their neighbourhood's choice** (13.7% more than 15 slices), against 2.3% over
the ant's body and 3.9% on the in-focus PCB. At kernel 51 the confetti coarsens
into blobs - 6.5% - but the seams get larger, not fewer.

**Fix.** A depth map is piecewise-smooth: it varies continuously across a
surface and steps only at an occlusion. So the incoherent pixels are outliers
against their own neighbourhood, which is exactly what a median removes while
leaving genuine steps standing. Three passes of a 5x5 median over the finished
index map, before a single pixel is gathered:

```python
best_index = _regularise_index(best_index)   # 3 x cv2.medianBlur(..., 5)
```

Where the measure has a real peak the argmax already agrees with its
neighbours and the median is a no-op there, so sharp detail is not what pays
for this.

**Why three narrow passes rather than one wide median.** Iterating a narrow
median converges towards its root signal - it removes what is smaller than the
window without eroding what is larger - whereas a single wide median rounds off
real depth features of its own size too. Measured on the ant stack's background
region, three 5x5 passes match a single true 9x9 on incoherence (seam density
0.0274 against 0.0259) while keeping *more* detail on the in-focus crops. It is
also the only affordable option: 5 is the widest window `cv2.medianBlur` takes
for anything larger than uint8, and the exact wide median is a scipy call - on
this 1.7 MP index map, 1.07 s for a 9x9 and 2.7 s for a 15x15, against 15 ms
for three cv2 passes.

**Rejected alternatives.** Guided-filter smoothing of the index map, gated or
not, cost a third to a half of the in-focus Laplacian variance and *raised*
incoherence,
because rounding a smoothed float index dithers. Confidence-gating the median -
applying it only where the winner fails to stand out - was worse than applying
it everywhere: the gate's own boundary becomes a seam.

**What the measurements said.** PSNR against the ground-truth reference, mean
over all seven scenarios, and SSIM:

| Kernel | PSNR before | PSNR after | SSIM before | SSIM after |
|---|---|---|---|---|
| 9 | 41.87 | **42.27** | 0.9693 | **0.9820** |
| 31 | 41.48 | **42.16** | 0.9771 | **0.9864** |
| 51 | 41.26 | **41.29** | 0.9858 | **0.9890** |

`deep_stack` - the scenario that models this exact failure - gains the most:
26.05 -> 27.81 dB and 0.806 -> 0.894 SSIM at kernel 9, with its defocus seam
excess halved (1.949 -> 0.969).

**The one real cost, located.** `long_stack` at kernel 51 loses 0.78 dB. That
scenario is 12 depth bands 27 px tall, so a 51 px pooling window is already
wider than a band; the median shifts those boundaries a couple of pixels
further. Splitting the extra squared error by row shows **100.5% of it lands
within 4 px of a band boundary** - band interiors come out very slightly
*better* (-0.02 MSE). It saturates at one pass, so fewer passes do not buy it
back.

**A metric moved the wrong way, and again should have.** `deep_stack`'s
`spatial_frequency` fell 14.48 -> 9.80. The reference's own spatial frequency
is **8.54**: the old render scored 69% *above* the truth because the metric
counts gradient energy without asking whether the reference has any there, and
the tearing is gradient energy. Split by region, background gradient went from
4.9x the reference's down to 2.7x while the subject moved -5%. `block_speckle`
rose 0.156 - one 8x8 cell out of 640, the metric's quantum, leaving the result
one block off the reference's own 2.188.

**A latent NaN came out with it.** The two pooled terms item 21 multiplies are
non-negative in exact arithmetic but not in float32 - `boxFilter` accumulates a
running sum, and over a window spanning a sharp edge the cancellation can leave
a tiny negative. Unclamped that reached `sqrt` as a NaN, and a NaN loses every
`>` it is compared with, so the pixel silently kept whichever frame happened to
be there. Clamped on both paths.

**Cost.** Three median passes over one index plane: ~15 ms on 1.7 MP. The
333-frame ant stack, 8 threads, best of three: 4.22 s against 4.15 s at kernel
9, 4.46 s against 4.49 s at kernel 51 - inside the run-to-run noise. On the GPU
the median runs on the host, over a plane that is a couple of megabytes, and
only the pixels it actually moved are re-gathered.

**Guarded by** the quality ratchet (re-recorded, `depthmap_max` only),
`tests/test_depthmap_halo.py`, `tests/test_depthmap_gpu.py`, and the CPU/GPU
parity pairs - MODE_MAX is bitwise identical between the two paths at kernels
9, 31 and 51 and for 16-bit frames with halo suppression, which the shared
`_regularise_index` is what makes possible.

---

## 23. Depth Map still hard-selects where it has no basis to select at all

**Method:** Depth Map (Max) | **Category:** Quality | **Impact:** High |
**Effort:** Medium | **Fixed in 1.35.0, corrected in 1.36.0**

Item 22 despeckled the index map and stopped there. It removed the confetti and
left the blobs: on the reference 333-frame ant stack the chosen index still
varies by **12.9 slices inside a 9x9 window** over the never-resolved
background, against 3.8 over the ant's body, because the incoherence is far
wider than the 5x5 median being run over it. What reaches the output is a mosaic
of hard-edged patches rather than confetti - coarser, and no less visible.

Helicon Focus renders the same region smoothly at *radius 1, smoothing 2* - its
lowest settings, where a hard argmax over a 1 px window would be pure noise.
That is the tell: Method B is not despeckling a hard selection. Nor is it
averaging, because the background it produces keeps the contrast and brightness
of everything around it.

**Fix.** Decide which pixels the focus measure can speak for, and replace the
depth of the ones it cannot with what their neighbourhood implies.

*Trust*, from the participation ratio of a pixel's energy across the stack,
`eff = (sum E)^2 / sum(E^2)` mapped onto `1 - (eff - 1)/(N - 1)`. The obvious
`(peak - mean)/peak` was rejected for drifting with stack depth - the same
background scores 0.69 at N=333 and 0.60 at N=15, so no fixed threshold means
the same thing twice - where the participation ratio holds that background at
**0.378-0.397** and the subject at **0.583-0.626** across N=15 to 333. It also
had to be absolute rather than a percentile of this image's own histogram:
tiled fusion runs each tile through independently, and a threshold read off the
tile would lay a seam along the tile lattice.

*A trust-weighted low-pass of the depth map*, carried on a Gaussian pyramid so
its reach grows by doubling rather than by window width, mixed back by trust.
The depth field it produces is continuous across the boundary between measured
and inferred, which is what keeps the seam out: only the depth changes there,
never the rule that renders it.

### What 1.35.0 got wrong

The first version of this varied the *rendering* by trust - it widened the slice
blend where trust was low, so unresolvable regions resolved to the local mean of
the stack. It scored well on every metric available and was visibly wrong on the
user's own renders, in two ways that share one cause.

**The trust map is bimodal, so "proportional to (1 - trust)" is a step.** On the
ant stack only **9-14% of pixels** land between 0.05 and 0.95; 34% at kernel 5
are fully untrusted and most of the rest fully trusted. A blend width gated on
that quantity does not ramp, it jumps - and drew exactly the kind of hard-edged
boundary the exercise was removing.

**Averaging tens of defocused, drifting frames converges on something flatter
and paler than any one of them.** The regions it covered came out as washed-out
patches that had *lost* texture the plain hard select still had - visible at
kernel 5 as flat polygons with a hard rim, and worst at `ds100 bl100`.

Both are fixed by leaving the render alone: one tent, `BLEND_WIDTH_FLOOR` wide,
for every pixel in the frame. `slice_blending` was removed rather than
re-defaulted, because it has no safe range - a uniform blend wider than a slice
softens resolved detail, and a trust-gated one steps.

### The weight floor, which is what makes the low-pass a low-pass

Weighting the pyramid by trust *alone* makes it an extrapolation: a region
nothing resolves takes its depth entirely from the nearest pixels that were
resolved. On `deep_stack` that renders the background at the subject's depth and
costs **7.1 dB** against the plain hard select, because the reference there is
frame 0 - a never-sharp background still has a real, weak preference of its own,
and discarding it is discarding data. Adding `DEPTH_FILL_WEIGHT_FLOOR` to every
pixel's weight lets such a region average its own measurement instead: the
scatter goes, the regional level stays. At the shipped three levels the floor is
worth **3.6 dB** (28.02 -> 31.64, measured at kernel 31).

**Interpolating between slices was tried and is also wrong.** Fitting a
sub-slice peak (parabola through the winning triple) and rendering through it
reads like the natural completion of a continuous depth map, and it cost **8 dB
on `long_stack`** (42.46 -> 34.50), where every pixel is genuinely resolved by
exactly one frame: the two frames either side of a peak are the two that resolve
a pixel *worst*, so mixing them veils detail the winner keeps.
`BLEND_WIDTH_FLOOR = 0.75` is what forbids it - under 1 so a whole-slice depth
puts the entire tent on one frame, over 0.5 so a depth landing midway still has
both neighbours inside it rather than neither.

**What the measurements said.** `deep_stack` is the only scenario that moves at
all; the other six are bit-identical, because trust is 0.91-0.999 across them
and the low-pass of an already coherent depth is that same depth:

| Metric (ratchet, `deep_stack`) | 1.34.0 hard select | 1.35.0 blend | 1.36.0 low-pass |
|---|---|---|---|
| psnr | 27.814 | 27.759 | **28.612** |
| ssim | 0.894 | **0.969** | 0.964 |
| block_speckle | 2.344 | **2.188** | **2.188** |
| defocus_seam_visible | 2.252 | 1.920 | **1.530** |
| defocus_seam_excess | 0.433 | **0.260** | 0.365 |
| seam_excess | 0.969 | **0.612** | 0.678 |
| qabf | 0.341 | **0.371** | 0.352 |

The last two versions score within a whisker of each other - 1.35.0 is even
ahead on three of the seven - which is the point worth recording: **the metrics
could not tell them apart, and the renders could.** No scenario in the suite has
a region that is both unresolvable and textured, so nothing here charges for
washing that texture away, and none of them is wide enough for a trust boundary
to fall in open background rather than against a frame edge. The check that
caught it was looking at the picture.

**A metric moved the wrong way, and should have.** `spatial_frequency` fell
9.80 -> 8.59, back towards the reference's own **8.54** that item 22 measured.
Split by region, the subject band is unchanged and the never-resolved background
is where the activity went - it was the mosaic's edges.

**Cost.** Full-resolution ant stack, 333 frames at 1212x1819, 8 threads: 6.8 s
before, 16.4 s after. The measurement pass is unchanged bar two running sums;
the second pass over the frames is what the coherent render costs, and it
replaces a gather that was already reading them.

**`depth_smoothing` at 0 is the old hard select, byte for byte** - asserted
against the pre-change implementation over seven scenarios at kernels 9 and 31,
MODE_AVERAGE included and untouched.

**Guarded by** the quality ratchet (re-recorded, `depthmap_max`/`deep_stack`
only), `tests/test_depthmap_halo.py` and `tests/test_depthmap_gpu.py`. The GPU
path does not reimplement any of this: every map involved is a single plane, so
they are settled by the host's own helpers and only the gather they drive stays
on the device, which makes the depth map identical between the paths by
construction.

---

## 24. Depth Map (Average) stops selecting as the stack deepens

**Method:** Depth Map (Average) | **Category:** Quality | **Impact:** High |
**Effort:** Low | **Fixed in 1.37.0**

The mode weighted every frame by its focus energy and added the results up:
`sum((E + b) * I) / sum(E + b)`. That is selective on a three-frame fixture and
it is not selective at all on a real stack, because a defocused frame does not
measure zero at a detailed pixel - it measures a small fraction of the peak -
and the sum adds that fraction up once per frame.

Measured on the reference 333-frame ant stack at kernel 21, over the frame:

| | |
|---|---|
| Defocused tail, summed over the other 332 frames | **50x** the winning frame's own energy |
| Share of the output the winning frame contributed | **1.8%** at the median pixel, 7.8% at p99 |
| Frames effectively mixed per pixel (participation ratio) | **233** of 333 at the median |

So the "contrast-weighted average" was, in practice, the arithmetic mean of 333
differently defocused frames. The mean of many defocus kernels is one enormous
defocus kernel: a veiled, low-contrast result with lifted blacks, bokeh from
every bright object smeared across the shadows beside it, and fine detail gone.
Against the same stack's Max render it held **27%** of the mean absolute
Laplacian (37.3 against 138.7) and **40%** of the local standard deviation.

It got worse the deeper the stack went, which is exactly why nothing caught it:
the selectivity of a linear weighting falls as `1/N`, and every scenario in
`tests/fusion_scenarios.py` bar `deep_stack` is 12 frames or fewer.

**Fix.** Weight by a *power* of the energy, `w = ((E + b) / s) ** p`, exposed as
a **Selectivity** dial (0-100, default 50 -> `p = 4.5`; 0 is the old linear rule).

Raising a *ratio* of energies to a power is what makes this depth-invariant.
Where one frame genuinely stands out, its weight pulls away from the rest as the
p-th power of how far it stands out, so 332 frames at a seventh of the peak can
no longer outvote it. Where every frame measures the same the ratios are all 1,
every weight comes out equal whatever `p` is, and the blend is still the plain
mean that buys the mode its multi-frame SNR. The dial therefore costs nothing in
the regions averaging is the right answer for.

On a controlled 120-frame stack - flat noisy half, single-frame-sharp half:

| p | Contrast kept in the sharp half | Frames averaged in the flat half |
|---|---|---|
| 1 (old) | 35% | 118 of 120 |
| 3 | 94% | 103 |
| **4.5 (default)** | **99%** | **85** |
| 8 (dial at 100) | 101% | 50 |

Detail saturates by `p = 4-6` while the flat-region averaging keeps eroding,
which is what puts the default at 4.5 and the ceiling at 8.

**On the project's own scenarios**, PSNR against the all-in-focus reference,
dial at 0 -> 50:

| Scenario | Frames | 0 | 50 | Max, for scale |
|---|---|---|---|---|
| `deep_stack` | 64 | 22.01 | **24.44** | 28.61 |
| `long_stack` | 12 | 30.55 | **38.41** | 45.41 |
| `sensor_noise` | 3 | 41.41 | **49.49** | 50.74 |
| `low_contrast` | 3 | 64.28 | **69.65** | 68.95 |
| `fine_texture` | 3 | 32.94 | **35.81** | 35.01 |
| `depth_edge` | 2 | 39.35 | 39.30 | 35.86 |
| `saturated_colour` | 3 | 33.15 | 32.42 | 32.22 |

Only `saturated_colour` gives ground, by 0.7 dB, and it gives it in the
direction of Max - a scene the metric prefers blended is exactly the scene a
more selective blend scores worse on. The synthetic photographic fixture moves
41.0 -> 51.3 dB.

**The energy scale now has to be known before the blend rather than after it.**
`b` was recoverable for free from the weight sum once the linear fold was done;
a power cannot wait. Only `b` survives the normalisation - the scale itself
cancels and is carried purely to keep the exponent on numbers of order one - and
`b` is a soft floor that tolerates being wrong by a factor of two, so it is
estimated from **8 evenly spaced probe frames** rather than a second full pass.
That lands within 3% of the true mean on the ant stack and, being a fixed count,
costs under 2% of a deep run.

**Cost: none measurable.** 333 frames at 1619x1064, 8 threads: 7 s before, 7 s
after. The exponent is taken by squaring rather than by `pow()` - the dial only
ever asks for half-integers, which is why it is quantised to them - and it is
taken on the measurement pool alongside the energy it belongs to. The
reduction sheds a whole float32 frame - the plain-sum accumulator the deferred
baseline needed is gone - and each in-flight measurement task gains two
single-channel planes for the squaring, which `_worker_bytes` now charges for so
the pool still sizes itself against what it actually holds.

### What it uncovered in the halo dial

The blend finally selecting made a second thing visible that had been there all
along. Halo suppression fills the band around a sharply focused region with that
frame's defocused pixels; where there was a glow that is the trade it is sold
on, and where there was not it is just a band of defocus - and the focus measure
cannot tell the two apart, so it does it around every contour in the frame. The
old linear blend averaged the result into mush, and mush hid it.

On the reference ant stack at kernel 5, share of the frame moved by more than 8
levels against the same render with the dial off:

| Radius | 2 | 5 | 9 | 15 |
|---|---|---|---|---|
| Frame affected | 8.5% | 18.3% | 24.8% | **29.9%** |

Clean, faint, obvious, bad - with nothing changing but the radius. MODE_MAX has
the same blobs at the same radii and always has; `depth_smoothing`'s low-pass
partly covers for them there, and MODE_AVERAGE has no equivalent stage.

This is the dial behaving as documented rather than a defect - `depthmap.py`
has said since 1.33.0 that it "does not remove a ring, it fills one with
defocused pixels, and it does so whether or not there was a glow to fight" - so
the response is a range rather than an algorithm change. The halo slider's
ceiling now follows the kernel (`constants.halo_radius_ceiling`), turning the
module's own "keep the radius well under the kernel size" from a sentence in a
comment into something the UI enforces, and bringing a radius dialled in at a
wide kernel down when the kernel narrows. The engine is left permissive: a
scripted caller with a genuinely wide glow can still ask for more, which is what
`tests/test_depthmap_halo.py` needs at r=8 on a fixture blurred by 31 px.

**Guarded by** `tests/test_depthmap_selectivity.py` - both halves of the bargain
(detail recovered on a deep stack, flat regions still averaged and still far
quieter than Max), the exponent helper against `np.power` at every setting the
dial can produce, `selectivity=0` against the linear rule rebuilt from its
definition, and MODE_MAX proved indifferent to the dial - plus CPU/GPU parity at
both ends of the dial in `tests/test_depthmap_gpu.py`.

---

## 25. Depth Map (Average) selects hard, and has nothing to make that coherent

**Method:** Depth Map (Average) | **Category:** Quality | **Impact:** High |
**Effort:** Medium | **Fixed in 1.38.0**

Item 24 stopped the blend collapsing into the mean, and in doing so turned it
into something it was never given machinery for. Measured on the reference
333-frame ant stack at kernel 9, selectivity 100:

| | |
|---|---|
| Share of the weight the winning frame takes, median resolved pixel | **96%** |
| Frames effectively mixed there (participation ratio) | **1.1** of 333 |

So at any selectivity high enough to keep the haze off a deep stack, MODE_AVERAGE
*is* a hard select - and it therefore inherits the hard select's failure without
inheriting MODE_MAX's cure for it (items 22 and 23). Local spread of the depth
the blend actually renders from, over a 9x9 window:

| Region | Detail | Mid | No frame resolves it |
|---|---:|---:|---:|
| Spread, in slices of 333 | 0.24 | 0.57 | **7.59** |

Quarter of a slice where the stack resolves something, and pixels side by side
drawn from frames eight apart where it does not. Those frames do not carry the
same local brightness - a defocused neighbour's glow has moved between them - so
the region comes out as blotches, which is what this mode has always done to
smooth dark surfaces.

The only setting that suppressed it was a pooling window wide enough to average
the incoherence away, and that window is also the thing deciding what counts as
detail. The kernel was doing two jobs and could not do both: `k51` was the
cleanest render of the previous sweep and it is soft, `k5` the sharpest and it
blotches.

**What did not work.** Two obvious fixes were measured and dropped, both
recorded here because each looks right on paper:

* **Raise the baseline** so flat regions average. It does exactly that - flat-zone
  high-pass energy falls from 2.19x Helicon's to 1.02x - but the baseline is an
  *absolute* level and a modest real focus peak in a dark region sits below it,
  so tile agreement falls from 0.945 to 0.906 as the detail goes with the grain.
* **Gate on trust**, the participation-ratio statistic MODE_MAX already uses. It
  fixes the flat regions outright (7.59 slices to 0.38) and tears the mid zone
  apart doing it (0.57 to **15.40**), because trust is bimodal - only 8.9% of
  pixels land between 0.05 and 0.95 - so a rule that switches on it is a step and
  not a ramp. This is the same failure `DEFAULT_DEPTH_SMOOTHING` documents from
  1.35.0, reproduced in the other mode.

**Fix.** Smooth the decision instead of the evidence. Each frame's *share* of the
blend is passed through a guided filter (He et al., the one `gff.py` already
carries) with that frame as its own guide, before the pixels are gathered -
exposed as a **coherence radius** in pixels, 0 being off and byte-for-byte the
old blend.

Three properties make it the right shape where the two attempts above were not.
It is applied uniformly to every pixel, so it draws no boundary: where the blend
already agrees with itself over a neighbourhood the share is locally constant and
filtering returns it unchanged. It runs *after* the exponent rather than before
it, so unlike a wider pool it cannot leak a sharp contour's energy onto the
background beside it - the ring item 21 removed. And guiding each share with its
own frame is what keeps it from costing detail: where that frame is sharp its
guide has edges and the share keeps them, and where it is defocused the guide is
featureless and the share is simply smoothed, which is precisely the case where
the choice was arbitrary.

The share, rather than the raw weight, is what has to be filtered - the raw
weight is an eighth power of an energy ratio spanning three decades, and a box
over that propagates the largest value in the window. Shares are bounded in
[0, 1] and sum to 1, so filtering them re-mixes a decision. That is also the cost:
the total must be known before any frame can be normalised, so the stack is
measured twice. Measured at 1.7 MP over 333 frames: **5.6 s to 10.3 s**.

On the reference stack against Helicon Focus method A radius 30 - its own
contrast-weighted average, and the render this mode is asked to come closer to -
at the shipped kernel and selectivity (9, 50):

| Radius | 0 | 4 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|---:|
| Fine-band energy, quietest fifth of the frame | 2.19x | 1.60x | 1.15x | **0.97x** | - |
| Fine-band energy, busiest fifth | 1.35x | 1.33x | 1.30x | 1.23x | - |
| Tile detail agreement | 0.945 | 0.963 | 0.972 | **0.977** | 0.972 |

0.977 is past every setting reachable without this - the best of the 31-render
sweep that preceded it was 0.967 - and inside the spread of Helicon's own three
methods against that render (0.963 to 0.988). It also unloads the kernel: with
the filter on, agreement at radius 16 is 0.976 at kernel 5, 0.977 at 9, 0.975 at
25 and 0.972 at 51, where without it the kernel had to be 51 to come close.

**Off by default, and the reason is the assumption.** The filter assumes what the
rest of the module assumes - depth piecewise-smooth, stepping only at an
occlusion - and it assumes it out to `radius`. The scenario stacks in
`tests/fusion_scenarios.py` are built the other way, one horizontal band per
slice, so a 320 px frame of 12 slices steps depth every 27 px:

| Scenario | coh 0 | coh 2 | coh 4 | coh 8 | coh 16 |
|---|---:|---:|---:|---:|---:|
| `fine_texture` | 35.81 | **37.75** | 36.11 | 32.89 | 29.83 |
| `long_stack` | **38.41** | 34.83 | 32.18 | 29.37 | 25.23 |
| `sensor_noise` | **49.49** | 46.09 | 43.73 | 41.01 | 38.14 |
| `deep_stack` | 24.44 | 24.48 | **24.48** | 24.33 | 23.91 |

Every one best at radius 0 to 4. They are not independent evidence of anything
else - they share one mask builder, and that geometry is the one this stage
cannot serve - but they are the geometry the suite guards, and short stacks do
not want the filter regardless: the haze it descends from is what a defocused
*majority* does to a blend, so a three-frame stack never had the disease. It
earns its keep on a deep capture of a real surface, which is the regime to reach
for it in.

**Guarded by** `tests/test_depthmap_coherence.py` - 0 reproducing the unfiltered
blend byte for byte, MODE_MAX proved indifferent, the filtered result held inside
the envelope of the frames it blended (a fitted linear model can return a
negative share, and a negative weight would subtract a frame), the filter itself
shown to leave a constant untouched and to follow a structured guide where it
smooths a featureless one, and the quality claim made where it is true: against
the bundled `samples/electronics_ant` capture and its Helicon render, since a
rendered stack's flat region is flat and its unfiltered blend already averages it
correctly. CPU/GPU parity in `tests/test_depthmap_gpu.py`.

**The harness could not have found this.** `focus_retention` is the sweep's
headline metric and it rewards whichever render kept the most local contrast -
which on a real capture is the one that kept the most grain. It ranked
`k5_sel100`, the worst match to Helicon of the 31, **first**, and `k51_sel25`,
the best, **eighth**. So `tests/render_matrix.py` gained a foreign reference:
`--reference-render` scores every candidate against another program's render of
the same capture, as `RefGap` and `RefAgree`. Neither is read pixel by pixel -
the two programs aligned the stack differently and no warp relates them - but
they are fitted by a similarity first, which is worth doing: tile agreement on
this capture reads 0.49 unfitted and 0.95 fitted, and the unfitted number ranks
misalignment rather than fusion.

---

## 26. Depth Map (Average) has no coherence along the frame axis

**Method:** Depth Map (Average) | **Category:** Quality | **Impact:** Medium |
**Effort:** Medium | **Fixed in 1.39.0**

Item 25's filter is edge-aware by design, which is what stops it costing detail -
and is also its ceiling. In exactly the regions that hold detail it follows the
guide and leaves the blend alone, so those pixels are still rendered from about
one frame. Measured against Helicon Focus method A radius 30, in fifths of the
frame ordered by how much detail *it* found there, the residual is the same
shape at every one of the 32 settings that sweep tried:

| quintile | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| best spatial setting (k25, sel 100, r 24) | 0.91 | 0.94 | 0.98 | 1.14 | 1.20 |

The quiet end can be put on Helicon's level and the busy end will not move - no
kernel, selectivity or radius brought quintile 5 below **1.18**. That is not a
tuning miss; it is the filter declining to act where it was told not to.

**Fix.** Pool each pixel's share of the blend over `slice_radius` slices either
side, before the spatial filter runs.

The frame axis is where the headroom was. This capture oversamples its own depth
of field about sevenfold - a real focus peak is 7 frames wide at half maximum,
and adjacent frames agree on the fine detail at a correlation of **0.92** - so
the frames around the winner resolve the pixel almost as well and differ mostly
in their grain. Averaging across that band divides the grain by roughly the
square root of the count at no cost in spatial resolution, which is the one thing
a spatial filter cannot offer.

The two compose, and they trade: the spatial radius moves the whole curve, the
slice radius pulls its busy end down hardest, and the pair levels it.

| Setting | RefGap | RefAgree | band 1..5 |
|---|---:|---:|---|
| k9 sel50 coh16 (item 25's best) | 0.157 | 0.9743 | 0.82 0.94 1.05 1.20 1.23 |
| k9 sel100 coh16 slice4 | 0.055 | 0.9736 | 0.91 0.98 1.00 1.06 1.06 |
| k9 sel100 coh13 slice5 | 0.029 | 0.9700 | 0.96 1.02 1.03 1.04 1.00 |
| **k25 sel100 coh16 slice5** | **0.016** | 0.9703 | 0.97 1.00 0.99 1.01 0.98 |

0.016 is within 3% of Helicon's own fine-detail energy in every fifth of the
frame. Agreement gives up 0.004 against item 25's best to get there, and stays
well clear of everything reachable before either stage existed (0.945-0.963).

The wide kernel winning is itself the point: with both stages supplying the
coherence, kernel 25 is free to be the more stable focus measure instead of
being the only way to get a clean render.

**It costs a pass and a half.** Both stages share the second measurement pass, so
having both on costs no more than either - but the spatial filter now runs in the
serial reduction rather than the measurement pool, because it has to follow the
slice pooling and that needs the neighbouring frames. On the 333-frame capture at
1.7 MP: 5.6 s unfiltered, 10.3 s with the spatial filter alone, **23.7 s** with
both.

**Off by default, and it depends on the registration.** Everything else in the
module re-mixes decisions *within* a frame; this averages different frames into
one pixel, so it is only free while they agree on where that pixel is:

| | registered | as captured |
|---|---|---|
| 111 frames, radius 1 | RefGap 0.160 -> **0.050** | detail recovered 1.16 -> **0.83** |

The value also follows the sampling rather than the stack - the same capture at
every third frame wants 1, not 5 - and overshooting is not subtle: at 8 slices
the whole curve drops to 0.77-0.81.

**Guarded by** `tests/test_depthmap_coherence.py` - 0 reproducing the unpooled
blend byte for byte, MODE_MAX indifferent, the radius clamped inside the stack,
the ring covering both ends of the stack on its shrinking window, the two stages
composing inside the envelope of the frames they blended, and the registration
dependence pinned on the unregistered capture. CPU/GPU parity in
`tests/test_depthmap_gpu.py`, where the device path buffers shares by frame index
so the window can straddle a chunk boundary.

**Why the claim is not made on a rendered stack.** Two things must hold at once
before this dial has anything to do: the blend must be concentrating on one
frame, and that frame's neighbours must carry nearly - but not exactly - the same
detail. Every fixture that can be built here fails one. A stack whose frames are
identical within a focus band measures the same energy across it, so the weights
are already equal and the blend already averages (36.7 dB at radius 0 against
34.4 at radius 1, all of it boundary spill). One with a smooth focus ramp has too
few frames for the exponent to concentrate at all. What produces the regime is a
deep sweep of real defocus, which is a capture and not a fixture.

---

## 27. Depth Map (Max) renders every pixel from exactly one frame

**Method:** Depth Map (Max) | **Category:** Quality | **Impact:** Medium |
**Effort:** Low | **Fixed in 1.42.0**

Items 21 to 23 gave this mode three stages, and every one of them acts on the
*depth field*: the two-scale measure decides it, the median despeckles it, the
trust map and fill repair it. None touches the rendering rule, and the rendering
rule is that a pixel comes from one frame whole. That is the mode's whole appeal
where the stack resolves - every output pixel is an input pixel, at the sharpness
the lens delivered - and it is also a ceiling nothing above it can lift.

Measured against Helicon Focus method B at radius 30, which is the same algorithm
by somebody who shipped it, over a 31-render sweep of kernel x depth smoothing x
halo on the 333-frame ant capture (`tests/render_matrix_suites.depthmap_max`):

- `focus_retention` is **1.000 in every row**, because a hard select cannot lose
  local contrast it was handed.
- grain where the picture holds no detail is **3.1** at the shipped defaults
  against the average mode's 1.5 to 1.9 on the same capture, and the only rows
  that reach the average's range are the ones smoothing the depth field flat.
- in fifths of the frame ordered by how much detail *Helicon* found there, the
  busiest fifth sits at **1.04 to 1.09** in all 31 rows - every kernel from 5 to
  51, every smoothing from 0 to 100, every halo radius.

That last number is item 26's finding mirrored. `depth_smoothing` is trust-gated,
so it acts hardest exactly where the measurement had nothing to say and declines
to act where it did: it takes the quiet fifth from 2.90 down to 1.19 and leaves
the busy one where it found it. No dial that moves the depth field can do better
there, because the pixel still comes from one frame and one frame's grain is what
it is.

**Fix.** Let `slice_radius` widen the tent `_gather_blended` already renders the
depth map through, from `BLEND_WIDTH_FLOOR` to `BLEND_WIDTH_FLOOR + radius`, so a
pixel is taken from the band of frames around its own depth rather than from the
single winner. It is item 26's dial, transplanted: this capture oversamples its
depth of field about sevenfold, so those neighbours resolve the pixel almost as
well and differ mostly in their grain, and averaging them divides that grain by
roughly the square root of the effective count at no cost in spatial resolution.

Swept against the smoothing dial rather than instead of it, because the two act
in opposite halves of the frame (`depthmap_max_slices`, kernel 25):

| Setting | RefGap | RefAgree | FlatNoise | band 1..5 |
|---|---:|---:|---:|---|
| ds50 slice0 (the shipped default) | 0.240 | 0.9622 | 3.15 | 1.52 1.27 1.20 1.13 1.07 |
| ds50 slice2 | 0.158 | 0.9610 | 2.88 | 1.31 1.19 1.14 1.06 1.01 |
| **ds50 slice4** | **0.108** | 0.9580 | 2.71 | 1.22 1.11 1.06 0.99 0.94 |
| ds50 slice5 | 0.097 | 0.9558 | 2.61 | 1.18 1.07 1.02 0.95 0.90 |
| ds50 slice8 | 0.162 | 0.9469 | 2.32 | 1.06 0.94 0.89 0.82 0.77 |

The gap more than halves for 0.004 of agreement, and the busy fifth finally moves
- 1.07 to 0.99 - which is the column no setting in the previous sweep could
touch. Radius 8 overshoots exactly as it does on the other mode: every fifth
lands under the reference at once and the gap climbs back.

**It costs nothing, which is the difference from item 26.** MODE_AVERAGE pays a
second measurement pass to form the shares its pooling reduces over. Here the
gather already walks the whole stack once the depth field is continuous, so a
wider tent asks the same frames for the same pixels with different weights:
11.7 s at slice 0 against 11.9 s at slice 4 on the 333-frame capture. The one
case that does change cost is `depth_smoothing 0` with a radius set, where the
mode leaves the plain copy for the tent gather - and that is the setting asking
for it.

**What it argues against, and why that is not a contradiction.**
`BLEND_WIDTH_FLOOR` has refused sub-slice blending since 1.35.0, on the grounds
that the frames either side of a peak are the two that resolve the pixel *worst*
among those that resolve it at all - and it cost 8 dB on `long_stack` to prove
it. That bounds the floor, not the dial above it: it is a statement about a stack
that steps a full depth of field per frame, which `long_stack` does and this
capture does not. The two coexist because the dial is off by default and is set
from the capture's sampling.

**It also corrects the previous sweep's own answer.** Read alone, that run says
to raise `depth_smoothing` to 100: gap 0.240 -> 0.086 at kernel 25. With the
frame axis available it is plainly the wrong route. ds100 buys the number by
flattening the depth field - agreement falls to 0.9477 against ds50's 0.9622, and
the crops show the ant's leg scales going with it - and once slice pooling is
also removing energy the two over-correct together, taking the gap back up to
0.173 at slice 2 and 0.376 at slice 8. `DEPTH_SMOOTHING_DEFAULT` stays at 50.

**Guarded by** `tests/test_depthmap_coherence.py` - radius 0 reproducing both of
MODE_MAX's rendering paths byte for byte, the grain falling while the textured
half both keeps its contrast and moves *closer* to the truth, and the overshoot
past the band pinned as a failure rather than left implied. CPU/GPU parity for
this dial and for `depth_smoothing` at both ends of its range in
`tests/test_depthmap_gpu.py`, plus a check that radius 0 leaves the device path
copying source pixels rather than gathering a tent.

**Why this claim *can* be made on a fixture.** Item 26 could not: the average
needs the blend to be concentrating on one frame before the dial has anything to
do, and a stack whose frames are identical within a focus band already averages
there. A hard select renders from one frame by definition, whatever the energies
look like - so the fixture that was useless for the average is precisely the one
that isolates the dial here.

---

## 28. Depth Map (Max) has no edge-aware stage, and its depth is noisy where the measure was merely adequate

**Method:** Depth Map (Max) | **Category:** Quality | **Impact:** Medium |
**Effort:** Low | **Fixed in 1.43.0**

Item 27 closed the RefGap and left RefAgree exactly where every earlier sweep
had: **0.958 to 0.962**, across 51 renders spanning every kernel, smoothing,
halo and slice setting. Two measurements say why, and neither is a dial set
wrong.

**Where the disagreement is.** Splitting the 32 px tiles by how much detail
Helicon found in each and correlating within each fifth:

| fifth of *their* detail | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| within-fifth correlation | 0.57 | 0.42 | 0.35 | 0.58 | 0.93 |

The busiest fifth tracks at 0.93. The middle tracks at 0.35. `depth_smoothing`
cannot reach that: it is gated on trust, so it acts hardest where the measure
said least and declines to act over exactly the regions the measure found
*merely adequate* - which is what the middle fifths are made of.

**It is not the registration either.** Sweeping that
(`tests/render_matrix_suites.depthmap_max_registration`, 6 settings on the full
capture) moves RefAgree by **0.002** for the stage that models focus breathing,
and **down 0.012** for a first-frame reference.

**Fix.** Pass the finished depth map through a guided filter, with the
all-in-focus picture as the guide - the stage MODE_AVERAGE was given at 1.38.0,
over the field this mode decides in. The guide costs nothing to obtain: whichever
frame is winning a pixel is the frame whose luminance belongs there, so it is
accumulated in the reduction that already finds the depth.

A depth map is piecewise-smooth *against the picture* - constant across a
surface, stepping where the picture shows an occlusion - which is exactly the
prior a guided filter encodes and exactly what an isotropic low-pass of the same
reach cannot honour without rounding the step off too.

Measured on the full capture at kernel 25 (`depthmap_max_coherence`):

| Setting | RefAgree | RefGap | FlatNoise | bands 1..5 |
|---|---:|---:|---:|---|
| ds50 (item 27's baseline) | 0.9624 | 0.241 | 3.15 | 1.53 1.26 1.20 1.13 1.07 |
| ds50 edge4 | 0.9693 | 0.206 | 2.94 | 1.46 1.20 1.16 1.10 1.06 |
| **ds50 edge8** | **0.9704** | 0.150 | 2.51 | 1.35 1.10 1.08 1.07 1.05 |
| ds50 edge8 slice2 | 0.9692 | **0.058** | 2.25 | 1.13 1.04 1.03 1.01 1.00 |
| ds50 edge16 | 0.9520 | 0.123 | 1.82 | 1.29 0.95 0.92 0.95 0.99 |

Both columns move together for the first time. Before this stage, slice pooling
bought RefGap by giving up RefAgree (item 27: 0.240 -> 0.108 cost 0.004); with
the depth field coherent first, `edge8 slice2` reaches a gap of **0.058** while
*still* scoring above every unfiltered row. Radius 16 is past the useful range
and below the unfiltered baseline, which is where `COHERENCE_RADIUS_MAX_DMAP`
puts the slider's ceiling at 12.

### What RefAgree cannot be pushed past, and why

**0.98 is not reachable on this metric, and the reason is not the fusion.**
`detail_agreement` correlates tile maps, so it only measures fusion once both
renders are on one geometry - and they never are. Three measurements bound it:

- **The residual.** After `fit_reference`'s similarity warp, our render sits
  **4.6 px rms** from Helicon's (median 2.2, p90 11.8, growing from 1.9 px at
  the centre of the frame to 6.0 px at the edge - the signature of focus
  breathing, which is a per-frame magnification no single warp removes). Two
  Helicon renders sit **0.5 px** apart, because they share an alignment.
- **What that costs.** Drifting a render by the measured field and scoring it
  against itself - zero content difference by construction - gives 0.9916 at
  block 32. Removing the same geometry from a real pair moves ours by
  **+0.004 to +0.009** and moves the already-aligned Helicon pair by **0.0000**,
  which is the control that says the procedure is not manufacturing agreement.
- **The ceiling.** Two settings of Helicon's *own* method B score **0.9872**
  against each other at that 0.5 px. Reading above 0.98 through a 4.6 px
  geometry difference would require our render to match `HF-B-30-1` more closely
  than Helicon's own `HF-B-1-2` does.

So the honest reading of this stage is: **0.962 -> 0.970 measured**, and
**0.966 -> 0.973 like-for-like** with the geometry removed. At 64 px tiles -
which `detail_map`'s own rule argues for at an offset this large, "a tile
several times wider than that offset" - the same render reads **0.9819** against
0.9750 unfiltered. That block size is not what the harness reports and has not
been changed here; it is quoted so the number is not mistaken for a ceiling of
the method rather than of the comparison.

**Off by default**, like the average's, and for the same reason: the reach is
the risk, and how far a depth field is smooth is a property of the scene.

**Guarded by** `tests/test_depthmap_coherence.py` - radius 0 reproducing both of
MODE_MAX's rendering paths byte for byte, the render staying inside the stack it
gathered from, and the property that makes it a guided filter rather than a blur:
a two-frame fixture whose correct depth steps down the middle of the picture,
where a radius of 12 must leave the step standing because the guide steps there
too. CPU/GPU parity in `tests/test_depthmap_gpu.py`, where the device builds the
guide in its own reduction and the host runs the filter, so the two agree on the
depth field exactly.

---

## Tuning Depth Map

What the two modes expose, what each dial was measured to do, and what is
deliberately left internal. Numbers are from the 333-frame 1.7 MP ant capture,
registered, scored against Helicon Focus method B for Max and method A for
Average - `RefGap` is how far our fine-detail energy sits from theirs (0 is
exact), `RefAgree` whether we found detail in the same places, and both are read
together because either alone is easy to satisfy.

**`kernel_size` (Kernel, default 9).** The window the focus measure is pooled
over. Since item 21 the measure is pooled at two scales and combined, so a wide
kernel no longer rings every subject and the dial is safe across its range: on
Max the gap moves from 0.261 at 9 to 0.240 at 25 and agreement from 0.9587 to
0.9622, which is small enough that the choice belongs to the picture. Small
follows fine detail and speckles on grain; large is a steadier measure and rounds
off narrow in-focus structures.

**`halo_radius` (Halo suppression, default off).** Item 24b's ceiling applies:
the slider stops at `min(kernel, 30)` because the radius is a claim on ground the
measurement has to be able to speak for. On this capture it buys little that the
two-scale measure has not already taken - 0.261 to 0.248 at kernel 9, radius 8 -
and past that it is filling a band around every contour with defocus. Reach for
it only when a glow visibly survives.

**`depth_smoothing` (Depth coherence, default 50, Max only).** Item 23's dial:
how readily a pixel's own decision is given up as unfounded, with the depth of
every pixel given up on interpolated from the ones that were not.

| setting | 0 | 25 | 50 | 75 | 100 |
|---|---:|---:|---:|---:|---:|
| RefGap (kernel 25) | 0.505 | 0.320 | 0.240 | 0.096 | 0.091 |
| RefAgree | 0.9486 | 0.9588 | 0.9619 | 0.9515 | 0.9477 |
| FlatNoise | 4.36 | 3.58 | 3.15 | 1.81 | 1.51 |

0 is the plain hard select and its torn background. The gap keeps falling to the
top of the range and agreement peaks at 50 and then goes with it, because past
that the fill is flattening depth structure that is real - the crops show it as
the subject losing its own surface texture. 50 is where those two part company,
and item 27 is why it stays there rather than following the gap.

**`selectivity` (Selectivity, default 50, Average only).** Item 24's dial, and
the one that makes the mode work at all on a deep stack. Below it the blend is
the arithmetic mean of every frame; every setting from 50 up is within a few
hundredths of a dB of the others on the report scenarios, and 100 is where item
26's sweep put all of its best rows once the coherence stages existed.

**`coherence_radius` (Weight coherence / Edge coherence, default off, both
modes).** Items 25 and 28: an edge-aware filter over the field each mode decides
in, before the pixels are gathered. The two want very different numbers, which is
why the slider is relabelled and re-ranged per mode.

| | Average (weight shares) | Max (depth map) |
|---|---|---|
| best on this capture | 16 | 8 |
| at the ceiling | 24 still gaining | 16 already below baseline |
| slider ceiling | 32 | 12 (`COHERENCE_RADIUS_MAX_DMAP`) |
| cost | a second pass over the stack | one plane, and the guide is free |

A weight share varies over the picture and tolerates being pooled a long way; a
depth field is nearly flat across a surface and steps at an occlusion, so past
roughly the pooling window the filter averages across the step instead of along
the surface. On the banded synthetic stacks in `tests/fusion_scenarios.py`, which
step depth every 27 px, the average wants **0 to 4** - the reach is the whole
risk in both modes, and the disagreement between fixture and capture is the scene
and not the number.

**`slice_radius` (Slice coherence, default off, both modes).** Items 26 and 27:
pooling along the frame axis, in the only form each mode can carry it. **4 to 5**
on this capture, which oversamples its depth of field about sevenfold; **0** on a
stack that steps a full depth of field per frame, where the neighbouring slices
are the ones that resolve the pixel worst. It follows the sampling and not the
scene - the same capture at every third frame wants 1, not 5 - and overshooting
is not subtle, with every fifth of the frame getting worse at once by 8 slices.
It is also the one dial in the module that depends on the registration, because
it is the only one that averages different frames into one pixel.

**`INDEX_MEDIAN_PASSES`, `INDEX_MEDIAN_KSIZE`, `NEAR_WINDOW_DIVISOR`,
`MEASURE_SIGMA`, `BLEND_WIDTH_FLOOR`, `TRUST_LO_MIN` and the fill's levels stay
internal.** Item 22's median and item 21's narrow window are corrections rather
than preferences - there is no picture that wants the ring back - and the trust
anchors are the `depth_smoothing` dial's own calibration, which is what the dial
slides. `BLEND_WIDTH_FLOOR` is the one a reader of item 27 will look for: it is
the floor `slice_radius` is added to, and lowering it would reintroduce the
sub-slice interpolation that cost 8 dB on `long_stack`.

Everything above is inherited by batch jobs from the main window, and anything
not at its default goes into the output filename, the way DCT's and the pyramid's
tuning already does.

---

## Tuning Pyramid

Everything item 19 added is exposed, because every one of them is a judgement
about a trade rather than a fixed truth. The defaults are the measured best on
the fixtures above; these are the reasons to move them.

**`energy_window` (Kernel slider, default 5).** The window each band's energy
is pooled over before frames are compared. The pyramid pools again at every
level, so the window at level k already covers 2**k times as much picture -
which is why the default is small where the depth map's is 9. Smaller follows
fine detail and speckles on grain (3 scores 49.8 dB on `fine_texture` against
46.0, and 29.5 on the veil against 35.1); larger decides regionally and rounds
off narrow in-focus structures (9 costs 10 dB on `fine_texture`). The slider
tops out at 51, where the pixel-domain methods' usefulness ends.

**`selectivity` (Selectivity, default Balanced = 8).** The headline control of
item 19. Average (2) and Soft (4) blend more of the stack into every band -
smoother backgrounds, softer real detail. Strict (32) and Winner takes all
(inf) approach the published rule; the last is the published rule, and brings
its stitched backgrounds with it (flat-field noise 1.39 against 0.63 on the
veil fixture). 2 costs 3-8 dB on the scenarios with real detail.

**16 was tried as the default in 1.44.0 and reverted.** It is worth writing
down, because everything short of the full capture says to make the change.
Re-measured against Helicon Focus method C on `electronics_ant` with the guard
widened to five ground-truth scenes, 16 gains **+1.08 dB** on `deep_stack`,
**+0.93** on `fine_texture`, **+0.71** on the veil and **+0.55** on
`long_stack` for 0.08 dB on `depth_edge`; on the 28-frame subsample the search
runs on, recovered tile contrast against `HF-C-1` goes 0.916 -> 0.947, which
reads as moving toward the reference.

On all 333 frames it reads 1.112 -> 1.233, which is moving *past* it, and out
of the bracket. Two things were wrong with the earlier reading:

* **Recovered contrast does not survive a change of stack density.** The module
  docstring says so about frame count and the autotune says so about quick
  runs; it applies just as much to the 28-frame fixture the search uses, where
  the crunchy half of the bracket cannot fire at all. A default chosen there
  has to be confirmed on the whole capture, which `confirm_at_full_density` in
  tests/fusion_autotune.py now does before anything is written.
* **The extra contrast is grain, not detail.** `band_ratios` splits fine-detail
  energy by how busy the reference found each region precisely to tell those
  apart. At 16 the render stands **4.02x** above `HF-C-1` where `HF-C-1`
  resolved nothing and **0.82x** where it resolved the most, against 3.78 and
  0.78 at the default: it is sharpening the grain faster than the picture.
  Flat-field noise goes 1.74 -> 2.02 and `reference_gap`, the pooled distance
  from the reference over all five zones, gets *worse* - 0.675 -> 0.710.

The ground-truth scenarios are not wrong, they are unrepresentative: their
defocus is a Gaussian the generator applied and their grain is not this
sensor's. Where the two disagree about a default, the photograph wins.

One further reading, recorded because it was nearly acted on:
`defocus_seam_visible` on `deep_stack` goes 2.70% -> 4.76% at 16. That is a
block-lattice measure applied to a method with no lattice, and it is not
monotone in this control - 2.0 scores 16.3%, 8.0 scores 2.76% and the published
choose-max 2.79% - so it is too noisy here to have carried the decision either
way. The full-density bracket is what carried it.

Also worth knowing: at *every* setting this method is far grainier than
Helicon's in the regions neither resolves - 3.78x in the quietest zone at the
shipped default - and slightly softer where the detail is (0.78x). That is a
real gap, it is not what this control fixes, and nothing in the suite currently
holds it. See reports/pyramid_reference_report.html.

**`coherence` (Scale coherence, default Off).** How much of a band's decision
comes from the coarser bands above it, so the bands of one frame decide
together: `S[i] = e[i] ** (1 - c) * up(S[i + 1]) ** c`, cascaded from the top,
so band i+k contributes with weight c**k. It is the one control that is off by
default. On these fixtures the envelope clamp already removes what it was there
to prevent, and a coarse-guided decision blurs the choice across a depth
boundary the coarse band cannot see: 0.5 costs 2.4 dB on `fine_texture` and
5.3 dB on `long_stack` for no measurable gain in flat-field noise. It is here
for the stack where the bands visibly disagree anyway - fine detail sitting on
a base that came from somewhere else - and the UI stops at Strong (0.75)
because 1.0 hands every band the coarsest band's decision and scores 17.3 dB on
`fine_texture`. Not a setting to render with; the end of a range.

**`base_selectivity` (Base band, default Balanced = 8).** How hard the coarse
base follows the frames that won the detail bands. Mean (0) is the pre-1.11.3
behaviour and the veil fixture shows why it went: 13.9 dB. Gentle (1) is item
16's weighting, 30.4 dB. Moderate (3) is 35.1 and was the default until the
2026.08.08 retune; 8 is 35.98 on the veil and 37.73 on `deep_stack` against
37.31, with `fine_texture`, `long_stack` and `depth_edge` unmoved to two
decimals and tile agreement with `HF-C-1` up rather than down. The gain is
small - it really has flattened by 5 - but it is a gain with nothing on the
other side of it, measured at every rung of the selectivity ladder, so the
default sits at the top of the curve instead of three-quarters of the way up.

**`noise_gate` (Ignore grain when nothing is sharp, default on).** Off is an
absolute comparison between frames, which is the published behaviour and what
lets a bright grainy frame win regions that hold no detail. Worth turning off
only to see what it is doing.

**`noise_percentile` (Grain estimate, default Balanced = 10).** What share of
each band that gate assumes holds nothing the band can resolve - the noise
level is read off the picture as this percentile of the band's own pooled
energies, so the number describes the frame rather than the sensor. Exposed in
the 2026.08.08 retune; before that it was a module constant the search could
reach and the panel could not, which made it the one dial of item 19 a user
could not act on.

Both ends are scene-dependent, which is the argument for a control. Minimal (2)
takes the estimate from whatever the single quietest corner is - a vignette, a
shadow - and so understates the grain over the rest of a frame that is not
evenly lit: on the ground-truth fixtures it costs 2.0 dB on `depth_edge` and
2.9 on `fine_texture`. Broad (20) reads it off pixels that are resolving
something and divides part of that signal away with the grain, which shows on
the veil fixture (-1.0 dB at the old base weighting; the current default
absorbs it). Between them the setting follows how much of the frame is subject:
Low (5) where the subject fills it, Broad (20) where it is mostly empty
background. Inert while the gate above is off, and the panel greys it out to
say so.

**`envelope` (Keep pixels within the source range, default on).** Off is what a
collapsed pyramid does unaided. Worth turning off only to measure the clamp's
effect, or if a stack legitimately needs values outside its own range, which a
focus stack does not.

**`levels` (Pyramid levels, default Auto = 5).** Auto means 5, clamped so the
coarsest band keeps both sides >= 2 px, which is what every render did before
the control existed. Fewer levels decide focus on coarser structure and can
miss fine in-focus detail; more separate scales finely and cost time.

**`NOISE_FLOOR_RATIO`, `NOISE_FLOOR` and the pyramid kernel stay internal.** The
first two are the guards on the gate's own arithmetic - what to do with a band
that has no measurable noise level at all - and have no meaning a user could
act on; the third is Burt-Adelson's.

All of it is inherited by batch jobs from the main window, and anything not at
its default goes into the output filename, the way DCT's tuning already does.

---

## Tuning DCT

What the method exposes, and what it deliberately does not. Measured on the
274-frame 2048x1364 stack from item 18. "Speckle" is the share of flat-area
blocks standing far outside their neighbours (`fusion_metrics.block_speckle`) -
the lone mismatched squares a block method produces, which the seam metrics
average away; the pyramid scores 0.103% on the same image.

**`kernel_size` (Smoothing slider).** Median-filters the focal-plane map. Since
item 18 this is the method's main quality control:

| kernel | speckle | background step | detail |
|---|---|---|---|
| 1 (off) | 0.385% | 0.236 | 16.32 |
| 7 | 0.333% | 0.286 | 16.13 |
| 31 (default) | 0.304% | 0.309 | 15.93 |
| 51 | 0.281% | 0.319 | 15.88 |
| 101 | 0.281% | 0.325 | 15.82 |
| 301 | 0.281% | 0.330 | 15.80 |

Two things follow. The default moved from 7 to 31: at 7 the lattice still shows.
And the ceiling moved from 51 to 151 - not because this image needs it (it is
flat from 51 onwards) but because the kernel counts *blocks*, so its reach as a
fraction of the frame shrinks as the frame grows. 51 spans a fifth of the map
here and a fourteenth of it on a 6000-wide stack, which is the size real macro
work runs at. The ceiling is per method (`_set_kernel_range` in `main.py`): the
pixel-domain methods that share the slider keep 51, where their own usefulness
ends.

Note the trade in the table: more smoothing removes speckle and costs a little
fine detail, because the map that stops flickering also stops following small
in-focus features. 31 is where that stops being worth it.

**`block_size` (DCT block).** Exposed at 1.18.0, having been fixed at 8 since
the method was written. It is the grid every decision is made on: 4 follows fine
detail and is noisier, 16 and 32 are steadier but step harder where near meets
far. `tests/fusion_registry.py` sweeps it 4 to 64.

**`plateau` (Focus tolerance).** Three presets - Crisp 0.85, Balanced 0.80,
Smooth 0.70 - and deliberately not a slider. It is the number behind item 18's
trade between a flat background and item 17's veil artefact, and the usable
range ends abruptly: 0.85 is clean, 0.90 fails outright. A raw control whose top
fifth reintroduces a fixed bug is a trap, so the value is clamped to 0.85 in
`dct.py` as well - the presets are the safe range, and the clamp is what
enforces it for any caller. Higher keeps a never-in-focus background nearer its
sharpest frame; lower averages more frames there, smoother and flatter.

**`blend` (Blend block seams).** On by default. Off restores the pre-1.17.1
behaviour of copying each block from a single frame, so every output pixel is
exactly some input pixel, at the cost of the lattice showing again
(`seam_excess` 0.683 -> 1.153 on `deep_stack`).

**`_POOL_WINDOW`, `_NOISE_PERCENTILE`, `_HIGHPASS_SCALE`, `_RUN_EXIT` and
`_SUBFRAME` stay internal.** All five are load-bearing for items 17, 18 and 4,
and none has a meaning a user could act on. The last two are the ones a reader
of item 4 will look for: `_RUN_EXIT` is where a run of in-focus frames is taken
to have ended, and its working range is bounded on one side by the veil artefact
and on the other by the defect it fixes; `_SUBFRAME` is the resolution the
focal-plane map is carried at, and halves are exactly what a run's midpoint
needs, so there is nothing to tune.

The three exposed controls are inherited by batch jobs from the main window, the
way the kernel and the registration checkboxes already are, and any that is not
at its default is written into the output filename so two renders that differ
only in tuning cannot collide.

---

## The quality ratchet

Every suite before this one asks "is this method broken", with thresholds far
below where the methods score. None of them asks "is this method worse than it
was", which is the question items 17 and 18 were both found by hand.

`tests/test_fusion_regression.py` records every method's scores on every
scenario in `tests/fusion_quality_baseline.json` and fails when one drops by
more than its tolerance. It also fails when a method scores *far above* its
baseline, because a stale baseline protects a quality level the code left behind
and the next regression slips under it. Re-record deliberately:

```bash
python -m tests.test_fusion_regression --update   # prints a diff first
```

Two things had to be added for it to be worth anything:

- **A metric that sees the artefact.** PSNR against a synthetic reference barely
  notices a block lattice. `fusion_metrics.block_seams` and `defocus_seams`
  compare the step across each block boundary with the typical step just inside
  the neighbouring blocks - content raises both, a seam raises only the first -
  and report severity and extent separately, because a hundred one-level steps
  and one hundred-level tear are the same number by any mean gradient ratio and
  only one of them is a defect.
- **A scenario in the failing regime.** The six report scenarios run 3 to 12
  frames; item 18 needs tens. `fusion_scenarios.deep_stack` is 64 frames with a
  background that is never sharp and drifts sideways as it defocuses.

Verified to bite: removing the noise normalisation (item 17) or the weight
compositing (item 18) both fail the ratchet, the latter on the seam metrics
across four scenarios.

**What it still does not catch.** `deep_stack` does not reproduce the reported
tearing - the old code scores *better* on it than the new one. 64 synthetic
frames of a smoothly drifting background do not put the old fallback in the
state that 274 real ones did. The check that caught item 18, and the only one
that confirms it fixed, is the reporter's own stack measured by hand. A fixture
that reproduces it would be worth having and does not exist yet.

`deep_stack` is deliberately kept out of `SCENARIOS`, so the characteristics
report and its tests still describe the six they were validated against. It is
worth a look on its own terms: on a stack this deep the guided filter fails to
beat a single frame (23.2 dB against 30.2), and StackMFF-V4 loses the colour
lead it holds everywhere else (18.5 levels of deviation, the worst of the field).
Neither is investigated here.

---

## Cross-cutting

**GPU is consistently worth it - and has since become the default.** Speed-ups
measured above are 2.6-8.6x for the guided filter and DCT, and 7.5-138x for
DTCWT. The GPU paths are verified to agree with their CPU twins by
`tests/test_fusion_quality.py::test_gpu_matches_cpu`, and as of 1.11.0 the
render pipeline requests them by default (`core/workers.py` constructs
`MultiFocusFusion(..., use_gpu=True)` with automatic CPU fallback).

**No method reports its confidence.** Every one produces a decision map it then
throws away. Surfacing "how sure was it here" - the margin between the best and
second-best frame - would let the UI flag regions where the stack simply lacks a
sharp frame, which is the most common cause of a disappointing render and is
currently invisible to the user. Still open at 1.11.0; the raw material now
exists in every method (`best_energy` in DCT, the depth map and pyramid, the
weight maps in GFF/GFG-FGF), but none of it is returned. Item 17 since gave DCT
the margin itself - `best_energy - second_energy`, already computed per block -
so for that method the metric now exists outright and only needs surfacing.

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
python -m tests.test_fusion_regression                         # every method, every metric
python -m tests.test_fusion_regression --update                # re-record the ratchet
```

Scenario definitions are in `tests/fusion_scenarios.py`, metrics in
`tests/fusion_metrics.py`, and the per-method calling conventions in
`tests/fusion_registry.py`.
