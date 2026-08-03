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
answer for the parts of a frame where nothing is in focus. The **Fixed** column
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
| 7 | DTCWT | Quality | Frames fused pairwise and recursively, so the result is order-dependent | Medium | Medium | 0% |
| 8 | DTCWT | Performance | CPU cost grows faster than image area | Medium | Medium | 25% |
| 9 | Guided Filter | Quality | The exposed kernel parameter barely changes anything - *investigated and closed in 1.5.6* | Low | Low | 100% |
| 10 | Guided Filter | Performance | Dead code; per-frame float32 copies dominate memory - *fixed in 1.5.6* | Low | Low | 100% |
| 11 | DCT | Quality | Crashes (default kernel) or corrupts indices on stacks of 256+ frames - *fixed in 1.11.2* | High | Low | 100% |
| 12 | DTCWT | Quality | Each colour channel picks its own source frame, so colour splits at depth edges | Medium | Low | 0% |
| 13 | DTCWT, Pyramid, Depth Map | Robustness | Folder loader copy-pasted four ways; mixed filenames crash the sort | Low | Low | 0% |
| 14 | DTCWT | Quality | Consistency vote biased toward the later frame at image borders | Low | Trivial | 0% |
| 15 | Depth Map | Quality | MODE_MAX decision map has no regularisation, so near-tie seams can speckle | Low | Low | 0% |
| 16 | DTCWT, Pyramid | Quality | Lowpass/base band fused by plain mean; ghosts under exposure drift - *fixed in 1.11.3* | Low | Medium | 100% |
| 17 | DCT | Quality | Focus measured as total block contrast, and grain deciding the rest, so blurred frames blank whole regions - *fixed in 1.15.1 and 1.15.2* | High | Low | 100% |
| 18 | DCT | Quality | Per-block winner never clears its margin in a deep stack, so 91% of blocks fell to a fallback that tore the background - *fixed in 1.17.1* | High | Medium | 100% |
| 19 | Pyramid | Quality | Choose-max stitches defocused regions out of frames that disagree, reconstructing filaments no frame had, and loses them to grain - *fixed in 1.19.0* | High | Medium | 100% |

**Overall: 70% done** - 13 of 19 items fully fixed, item 8 partially (the GPU
default shipped; the CPU cost itself is untouched), 5 untouched. Since 1.17.1 a
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

**Status at 1.11.0: open.** `dtcwt_torch.py` replicates the pairwise order
deliberately ("matches the sequential pairwise fusion order exactly"), so the
joint-selection rewrite must change both paths in the same commit.

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

**Category: quality. Impact: medium. Effort: low.**

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

**Category: quality. Impact: low. Effort: trivial.**

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
veil fixture). 8 and 16 are within a few tenths of a dB of each other on every
scenario; 2 costs 3-8 dB on the ones with real detail.

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

**`base_selectivity` (Base band, default Balanced = 3).** How hard the coarse
base follows the frames that won the detail bands. Mean (0) is the pre-1.11.3
behaviour and the veil fixture shows why it went: 13.9 dB. Gentle (1) is item
16's weighting, 30.4 dB. Balanced (3) is 35.1, Strong (8) is 35.9 and the gain
has flattened; against that, Strong costs 0.2 dB on `saturated_colour`. Nothing
between Gentle and Strong moves the other six scenarios by more than 0.1 dB.

**`noise_gate` (Ignore grain when nothing is sharp, default on).** Off is an
absolute comparison between frames, which is the published behaviour and what
lets a bright grainy frame win regions that hold no detail. Worth turning off
only to see what it is doing.

**`envelope` (Keep pixels within the source range, default on).** Off is what a
collapsed pyramid does unaided. Worth turning off only to measure the clamp's
effect, or if a stack legitimately needs values outside its own range, which a
focus stack does not.

**`levels` (Pyramid levels, default Auto = 5).** Auto means 5, clamped so the
coarsest band keeps both sides >= 2 px, which is what every render did before
the control existed. Fewer levels decide focus on coarser structure and can
miss fine in-focus detail; more separate scales finely and cost time.

**`NOISE_PERCENTILE`, `NOISE_FLOOR_RATIO` and the pyramid kernel stay
internal.** The first two are the gate's own calibration and have no meaning a
user could act on; the third is Burt-Adelson's.

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
