# Registration algorithms: what could be improved

An audit of the three alignment stages - `scale`, `homography` and `ecc` - in
the same form as [ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md) does for
fusion, and to the same standard: every claim here is measured, and the
measurement is given so it can be repeated or disputed.

Registration is measured differently from fusion, and better. The fusion audit
compares a fused picture against an all-in-focus reference and reads the
difference in dB. Registration has **exact geometric ground truth**: the two
handheld sample scenes record in `scene.json` the 2x3 affine that was applied to
every frame, so the error of an estimated alignment is a number of pixels rather
than an inference from the picture. A point `p` of the scene lands at `A_i(p)`
in frame i; registration estimates `H_i` mapping frame i onto the reference
canvas, so the point ends up at `H_i(A_i(p))`. Perfect registration makes
`H_i . A_i` the same map for every frame, so

    residual_i = RMS over p of | H_i(A_i(p)) - H_ref(A_ref(p)) |

is the misregistration in output pixels, independent of cropping. The `H_i` are
read back out of the running code rather than reimplemented - see
[Reproducing these measurements](#reproducing-these-measurements).

Measured on OpenFocus 1.30.1, one machine: Python 3.14, OpenCV 5.0, CuPy 14.1.1,
RTX 4080; the fixes since then were re-measured on the same machine at the
version each is dated to. Read absolute timings as relative. The pixel errors
and the dB were hardware-independent except where a row says GPU, and as of
1.30.5 they are hardware-independent everywhere - the GPU rows exist only in the
history of items 1, 3 and 4, which is where the second warp path lived and died.

Ranked by expected value: **impact** is how much it changes a real render,
**effort** is rough implementation cost.

---

## Summary

| # | Stage | Category | Issue | Impact | Effort | Fixed |
|---|-------|----------|-------|--------|--------|-------|
| 1 | ECC | Correctness | The GPU warp applies the inverse of the transform ECC measured, so on any CUDA machine ECC roughly *doubles* misalignment - 11.4 dB off the fused result | Critical | Trivial | **100%** |
| 2 | Homography | Quality | The 8-DOF fit is less accurate than not registering at all on both handheld samples; the two extra degrees of freedom fit nothing but noise - *fixed in 1.30.3* | High | Medium | **100%** |
| 3 | all | Quality | The reference frame is the only frame never resampled, so the focus measure sources 4-21x more of the picture from it than it should - *fixed in 1.30.4* | High | Low | **100%** |
| 4 | scale, ECC | Quality | GPU warping is bilinear where CPU warping is Lanczos4: 44% of the high-frequency energy lost, for almost no speed - *fixed in 1.30.5* | High | Low | **100%** |
| 5 | all | Robustness | `downscale_width` is silently overridden to 1024 for any frame >= 2048 px, i.e. for every real camera file - the exposed setting does nothing - *fixed in 1.30.6* | Medium | Trivial | **100%** |
| 6 | all | Quality | The reference frame defaults to `first`, though `middle` is better on every pipeline and every scene measured, and keeps more pixels | Medium | Trivial | 0% |
| 7 | all | Quality | Each stage is its own resample and its own crop, so the three-stage pipeline interpolates three times and throws away a further 10% of the frame | Medium | Medium | 0% |
| 8 | Homography, scale | Quality | Transforms are accepted on 6 matches with the RANSAC inlier mask discarded and no sanity check, and a bad one corrupts the whole chain after it | Medium | Low | 0% |
| 9 | ECC | Robustness | Crashes outright on single-channel input, which `scale` and `homography` both handle | Low | Trivial | 0% |
| 10 | scale, ECC | Performance | The GPU warp helper allocates 21x the frame size, regardless of frame size, with no fallback if that fails - *removed with the warp in 1.30.5* | Low | Low | **100%** |
| 11 | - | Maintenance | `_stabilisation_impl` is 117 lines of unreachable code | Low | Trivial | 0% |
| 12 | all | Robustness | Three copy-pasted folder loaders that disagree with each other | Low | Low | 0% |

**Overall: 6 of 12 done.** Item 1 is fixed in 1.30.2, item 2 in 1.30.3, item 3
in 1.30.4, item 4 in 1.30.5 with item 10, and item 5 in 1.30.6. That closes the
GPU-path defects: there is one warp path now, so registration produces the same
pixels on every machine and the "device" columns this document used to carry
have nothing left to compare. It also closes the one item where the code
ignored the user - `downscale_width` is now honoured at every frame size, which
is worth up to 1.75x of geometric accuracy on a full-size stack for anyone
willing to pay the time. What remains is a default worth changing (item 6) and
the structural one, item 7.

### The measurement everything else follows from

Geometric error of each pipeline against ground truth, reference frame `first`,
`--downscale-width 1024`. "gain" is how many times better than not registering
at all; below 1.00x means the stage made the stack *worse*. The "was" columns
are the pre-1.30.3 reading, before item 2.

| pipeline | handheld_drift mean px | gain | was | flower01_handheld mean px | gain | was |
|---|---|---|---|---|---|---|
| *unregistered* | 3.95 | - | - | 6.81 | - | - |
| **scale** | **2.22** | 1.78x | 2.22 | **2.57** | 2.65x | 2.57 |
| homography | 2.22 | 1.78x | **5.00** | 2.57 | 2.65x | **7.71** |
| **ecc** | **1.67** | **2.36x** | 1.67 | 6.40 | 1.06x | 6.40 |
| both (hom+ecc) | 1.69 | 2.34x | 1.57 | 6.50 | 1.05x | 7.07 |
| **scale+hom** (app default) | 2.23 | 1.77x | **3.98** | **2.06** | **3.30x** | 3.37 |
| scale+ecc | 1.69 | 2.34x | 1.69 | 6.50 | 1.05x | 6.49 |
| scale+hom+ecc | 1.63 | 2.42x | 1.58 | 6.63 | 1.03x | 6.77 |

Item 3 (1.30.4) left every single-stage row of this table bit-identical, and
moves only the two-stage `scale+hom`: 2.31 -> 2.23 px and 1.50 -> 2.06 px, since
the second stage now detects its features on a resampled anchor frame. That is
the only accuracy this document trades for item 3, and it is discussed there.

No pipeline is now worse than doing nothing on either scene. The two stages that
fit a constrained model where the motion is constrained - `scale`, and
`homography` since 1.30.3 - land in the same place on both scenes, as they
should, since they now differ only in what they do when the perspective is real.

---

## 1. ECC's GPU warp applies the inverse of the transform it measured

**Category: correctness. Impact: critical. Effort: trivial. Fixed in 1.30.2.**

`cv2.warpPerspective(img, M, dsize)` computes `dst(x, y) = src(M⁻¹ · (x, y))` -
it takes the *forward* source-to-destination map and inverts it internally.
`map_coordinates` has no such convention: it needs the destination-to-source map
handed to it explicitly.

The `scale` stage knows this. `_warp_perspective_gpu` inverts first
(`core/registration.py:122`), and its docstring says why:

```python
M_inv = np.linalg.inv(np.asarray(M, dtype=np.float64))
...
src = cp.asarray(M_inv) @ coords
```

The ECC stage has its own copy of the same code, inlined, and it does not
(`core/registration.py:936`):

```python
H_gpu = cp.asarray(H_final)
src_coords_homo = cp.matmul(H_gpu, coords)
```

The comment above it - "H_final is already H_inv, i.e. the mapping from
destination to source" - is the mistake in words. `H_matrices[i]` is
`inv(H_global)`, which maps frame i onto the frame-0 canvas: that is the
*forward* map, which is exactly why the CPU branch four lines below hands it
straight to `cv2.warpPerspective` with no `WARP_INVERSE_MAP` flag. Both branches
cannot be right, and the GPU one is not.

So every frame is warped by the inverse of its correction. Instead of removing
the measured displacement, ECC applies it a second time in the opposite
direction. On a stack drifting 2.5 px per frame:

| frame | true offset | after ECC, CPU | after ECC, GPU |
|---|---|---|---|
| 1 | 3.08 px | 0.007 px | 6.24 px |
| 2 | 6.16 px | 0.004 px | 12.17 px |
| 3 | 9.24 px | 0.008 px | 18.64 px |
| 4 | 12.32 px | 0.020 px | 24.57 px |
| 5 | 15.40 px | 0.017 px | 30.83 px |

The CPU path is excellent - ECC is a sub-pixel method and it lands within
0.02 px. The GPU path doubles the error it was asked to remove.

End to end, on the fused picture (Pyramid, against each scene's own all-in-focus
ground truth):

| scene | pipeline | CPU | GPU | cost |
|---|---|---|---|---|
| handheld_drift | ecc | 30.75 dB | 19.32 dB | **-11.43 dB** |
| handheld_drift | both (hom+ecc) | 30.53 dB | 20.33 dB | **-10.20 dB** |
| handheld_drift | scale+hom+ecc | 30.67 dB | 20.25 dB | **-10.42 dB** |
| flower01_handheld | ecc | 27.86 dB | 18.13 dB | **-9.73 dB** |
| flower01_handheld | scale+ecc | 27.85 dB | 23.81 dB | -4.04 dB |

For scale, an 11 dB loss is roughly four times the total spread between the best
and worst fusion method on a well-aligned stack. It also explains a second
symptom: on 16-bit input the GPU ECC path returns 40,269 pixels at zero where
the CPU path returns none - content pushed off the canvas by the doubled
displacement, appearing as a black border.

`homography` is unaffected because it has no GPU branch at all, which is why its
GPU and CPU rows are bit-identical throughout this document.

Worth noting where this lands in the app: `ROIAlignWorker` (`core/workers.py:96`)
hardcodes `method="ecc"`, so the ROI preview alignment is hit by this too.

### The fix

The inlined block was deleted and replaced by a call to `_warp_perspective_gpu`,
which already existed, already inverted, and already handled single-channel
input; the duplicated CuPy probe became the existing `_init_gpu_warp`. That
removed ~75 lines and left one GPU warp in the file rather than two that
disagreed. Re-measuring, every ECC pipeline now agrees with its own CPU path:

| pipeline | geometric error, GPU | CPU | was, GPU |
|---|---|---|---|
| ecc | 1.67 px | 1.67 px | 3.34 px |
| both (hom+ecc) | 1.57 px | 1.57 px | ~3.1 px |
| scale+ecc | 1.67 px | 1.69 px | ~3.3 px |

and on the fused picture the 11 dB is back:

| scene | pipeline | GPU before | GPU after | CPU |
|---|---|---|---|---|
| handheld_drift | ecc | 19.32 dB | **30.93 dB** | 30.75 dB |
| handheld_drift | both (hom+ecc) | 20.33 dB | **30.72 dB** | 30.53 dB |
| handheld_drift | scale+hom+ecc | 20.25 dB | **30.61 dB** | 30.67 dB |

The 16-bit black border went with it: the border pixels the GPU path used to
leave are now zero on both devices.

`tests/test_registration_gpu_warp.py` was the regression guard the audit asked
for - it ran a stage with the CuPy path forced on and off and asserted the two
agreed to within a fraction of a pixel, and separately asserted that registering
a stack with known drift *reduces* that drift rather than doubling it, which is
what separates this fix from its own reintroduction. Reinstating the sign error
moved frame 1 from 3.50 px of drift to 6.98 px and failed both. Item 4 deleted
the path the first half compared against, so the file is now
`tests/test_registration_warp_kernel.py`; the drift assertion is kept unchanged,
since a warp that applies its transform backwards is not a device-specific
mistake.

The GPU and CPU rows for `scale+hom` still differed slightly after this fix
(5.07 px against 3.98 px on handheld_drift). That was item 4, not this one:
bilinear warping changes the pixels the *next* stage detects features on, so the
two devices estimated marginally different transforms. It was the only GPU/CPU
divergence left, and it closed in 1.30.5 when the bilinear path was deleted.

One casualty worth recording. `test_a_registered_stack_with_a_black_border_stays_finite`
in `tests/test_pyramid_electronics_ant.py` built its fixture - a region exactly
zero in every frame - by running `both` registration on a real capture, which
only ever produced that border *because of this defect*, and only on CUDA. With
the warp fixed there is no border on either device, and masking one in by hand
does not reproduce the float32 residue, because the cancellation depended on the
pixels the bad warp produced. The test caught this itself: it asserts its own
fixture is valid before trusting anything. The pyramid guard it protects
(`np.maximum(pooled, 0.0)` in `_band_energy`) is still correct and still there,
so the test was retargeted at that invariant directly, with the box filter made
to return the residue the real one once did.

---

## 2. The homography stage is less accurate than not registering at all

**Category: quality. Impact: high. Effort: medium. Fixed in 1.30.3.**

`homography` and `scale` share their entire front end - SIFT on the same
downscaled frames, `BFMatcher` with the same 0.70 ratio test, RANSAC at the same
5.0 px threshold, the same neighbour-to-neighbour chain, the same crop. The only
difference between them is one line:

```python
H_local, mask = cv2.findHomography(pts_curr, pts_last, cv2.RANSAC, 5.0)     # 8 DOF
M_local, _mask = cv2.estimateAffinePartial2D(pts_curr, pts_last, ...)       # 4 DOF
```

That single line is worth a factor of two. Fitting both models to the *identical*
match set, pair by pair, and comparing each against the true pair transform:

| scene | 8-DOF homography | 6-DOF affine | 4-DOF similarity |
|---|---|---|---|
| handheld_drift | 1.364 px | 0.905 px | **0.679 px** |
| flower01_handheld | 1.401 px | 0.674 px | **0.583 px** |

The mechanism is visible in the estimated matrices. The two extra degrees of
freedom are perspective terms, and the true motion has no perspective in it at
all, so whatever they fit is noise. Expressed as the keystone they introduce
across the frame:

| pair | good matches | homography error | similarity error | keystone the homography adds |
|---|---|---|---|---|
| 3 | 27 | 4.455 px | 1.033 px | 4.08% of frame |
| 4 | 18 | 2.980 px | 1.179 px | 2.63% of frame |
| 5 | 34 | 2.219 px | 0.835 px | 2.12% of frame |
| 12 | 49 | 0.390 px | 0.155 px | 0.32% of frame |

Per-pair errors of ~1.4 px chain up over fifteen pairs, which is how a stage
that looks locally reasonable ends at 5.00 px and 7.71 px - worse than the 3.95
px and 6.81 px of leaving the stack alone. Adding it after `scale` degrades
`scale`'s own result: 2.22 px becomes 3.98 px on handheld_drift, 2.57 px becomes
3.37 px on flower01_handheld. **`scale` alone beats the app's default
`scale`+`homography` pipeline on both scenes.**

**The honest caveat:** these two scenes were generated from a per-frame affine,
so a homography cannot beat a similarity on them by construction. That is not a
rigged test, it is the physics of the instrument. A focus rail moves the focal
plane along the optical axis; the resulting frame-to-frame motion is a uniform
magnification about that axis plus small rotation and recentring - 4 DOF, which
is precisely the argument `_align_scale_impl`'s own docstring makes for fitting a
constrained similarity, and precisely the argument against the stage next to it
fitting eight. A handheld shot adds translation and roll, still within the
similarity. Genuine perspective needs the camera to *tilt* between frames, which
a stacking rig exists to prevent.

Three ways to spend the effort, cheapest first:

- **Demote it.** Make `scale` the default and `homography` opt-in for the cases
  that genuinely need perspective. Nearly free, and it is what the measurements
  support today.
- **Constrain it.** Reject an estimated `H_local` whose perspective terms imply
  more than a fraction of a percent of keystone, falling back to the similarity.
  This keeps the stage useful where perspective is real.
- **Fit the right model per stack.** Estimate similarity, affine and homography,
  and keep the one with the best cross-validated reprojection error over held-out
  matches. Most principled, most work.

### Fixed in 1.30.3, by the third route - and the second does not work

The keystone threshold was tried first and abandoned on arithmetic. Perspective
terms produce a keystone of roughly `theta * w / f` across a frame, so on a
normal lens a genuine 1-degree tilt bends the frame by about 2% - the same order
as the 2.1-5.7% the estimator was inventing out of noise on handheld_drift. A
threshold low enough to catch the noise (0.2%, which is where the measured
accuracy plateaus) rejects every real tilt down to a tenth of a degree, which is
the first option wearing the second's clothes. The size of the keystone does not
say where it came from.

What does say is whether the two extra degrees of freedom **predict matches they
were not fitted to**. Per pair, `_select_pair_transform` now:

- fits both models under RANSAC, as before for the homography and exactly as
  `scale` does for the similarity, and keeps the union of the two inlier sets -
  the union rather than the intersection because a match only the homography
  accepts is what real perspective looks like at the frame edges;
- splits those inliers 70/30 nine times, fits each model to the 70 and takes the
  median reprojection error on the 30. The fits inside the loop are plain least
  squares (DLT for the homography, a four-parameter normal equation for the
  similarity, written out in `_fit_similarity_ls`) so the comparison is
  deterministic - `estimateAffinePartial2D` is RANSAC-driven and would not be;
- keeps the homography only if it beats the similarity by 20% out of sample, and
  only on pairs with at least 24 inliers - three per degree of freedom, below
  which 8 DOF reproduce their own training points however wrong they are and
  validation cannot see the overfit.

Neither of the two constants is on a slope. The measured error/similarity ratio
is 0.13-0.62 on stacks with a real tilt and 0.92-1.39 on the two handheld
scenes, so the 0.8 threshold sits in the gap rather than inside either
population; sweeping it over 0.7-0.9, the inlier floor over 18-30 and the split
count over 3-15 moves the end-to-end result by at most 0.3 px.

**On the scenes with geometric truth**, the stage goes from the worst in the
document to level with `scale`, and keeps more of the frame because there is no
keystone to crop away:

| scene | pipeline | before | after | kept before | kept after |
|---|---|---|---|---|---|
| handheld_drift | homography | 5.00 px (0.79x) | **2.22 px (1.78x)** | 81.1% | **87.3%** |
| handheld_drift | scale+hom | 3.98 px (0.99x) | **2.31 px (1.71x)** | 81.6% | 84.5% |
| flower01_handheld | homography | 7.71 px (0.88x) | **2.57 px (2.65x)** | 91.2% | 93.2% |
| flower01_handheld | scale+hom | 3.37 px (2.02x) | **1.50 px (4.53x)** | 88.3% | 90.2% |

Every pair takes the similarity - 15 of 15 and 13 of 13 - which is the right
answer on scenes generated from a per-frame affine. On the fused picture (Pyramid, against each scene's own all-in-focus
ground truth) that is worth **+1.55 dB** and **+1.45 dB** for `homography`
alone, and +1.26 dB and +0.66 dB for the `scale`+`homography` default.

**Where the perspective is real the stage still fits it**, which is the half of
the change the accuracy table cannot show. On a stack whose frames differ by a
true camera tilt (`K R K^-1`, so no similarity can express it), before and after
are identical, while forcing the constrained model - the "demote it" option -
gives up a factor of 20:

| tilt per frame | keystone per pair | unregistered | `scale` (similarity) | homography, before | **after** |
|---|---|---|---|---|---|
| 0.15 deg | 0.22% | 15.43 px | 1.01 px | 0.15 px | **0.15 px** |
| 0.35 deg | 0.51% | 36.02 px | 2.42 px | 0.13 px | **0.13 px** |
| 1.0 deg | 1.45% | 103.48 px | 7.09 px | 0.23 px | **0.23 px** |

**Two places it costs a little.** `both` and `scale+hom+ecc` on handheld_drift
go from 1.57 to 1.69 px and 1.58 to 1.66 px: ECC was cleaning up after the
homography, and it now starts from a differently-biased estimate. Both keep 4-9
points more of the frame in exchange (76.1% to 84.7%, 77.7% to 81.7%), and the
fused result is unchanged to within 0.1 dB.

**Cost.** Ten extra least-squares fits per pair, on a few dozen points: 0.34 s
to 0.35 s for 16 frames of handheld_drift, i.e. nothing measurable.

**Guarded by** `tests/test_registration_homography_model.py`, which fails in
both directions - the four ground-truth assertions fail on the pre-1.30.3
estimator, and the two perspective assertions fail if the stage is changed to
always fit a similarity.

---

## 3. The reference frame is the only frame never resampled

**Category: quality. Impact: high. Effort: low. Fixed in 1.30.4.**

After `_rereference_transforms`, the reference frame's transform is exactly the
identity, and the crop translation folded in afterwards is an integer offset. A
warp by an integer translation is a copy: every interpolation weight collapses
to 1. So the reference frame comes out of registration byte-exact, and every
other frame comes out resampled once.

That is a systematic, artificial sharpness advantage handed to one frame - and
selection-based fusion decides everything by asking which frame is locally
sharpest. Measured on a strip of `handheld_drift` that is defocused in *every*
frame, so no difference there can be real focus:

| | reference frame 0 | mean of the other 15 | advantage |
|---|---|---|---|
| CPU warp (Lanczos4) | 43.4 | 34.8 | **+25%** |
| GPU warp (bilinear) | 43.4 | 15.2 | **+187%** |

And it changes the answer. Taking the per-pixel argmax of a pooled Laplacian
focus measure - the decision `depthmap`, `pyramid` and the rest all make - and
comparing against the scene's own `focus_index.png`:

| pipeline | handheld_drift | flower01_handheld |
|---|---|---|
| *ground truth: frame 0 is genuinely sharpest on* | *2.9%* | *0.0%* |
| unregistered | 5.1% | 4.7% |
| scale, CPU | 11.7% | 14.7% |
| homography, CPU | 11.4% | 15.7% |
| **scale+hom (app default), CPU** | **22.6%** | **29.2%** |
| scale, GPU | 53.6% | 57.2% |
| **scale+hom (app default), GPU** | **59.8%** | **67.8%** |

On a CUDA machine, **two thirds of the picture is being sourced from whichever
frame happened to be the alignment anchor**, on a scene where that frame is
genuinely the sharpest nowhere. This is the same failure
[ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md) §19 describes for the
pyramid - a selection rule with no answer where nothing is in focus - except
that here registration is *manufacturing* the tie-break rather than the fusion
method mishandling one. It also compounds with item 7: each additional stage
resamples the non-reference frames again while the reference stays untouched, so
the bias grows with pipeline length (23.6% for `both`, 34.9% for
`scale+hom+ecc`, CPU).

Everything above this line is the pre-1.30.4 measurement, kept as the statement
of the defect; the fix and its numbers follow.

**The fix is to give the reference frame the same treatment as everything else**
- put it through one identical resample rather than special-casing it into a
copy. It costs one warp. Simulated by pushing frame 0 through a half-pixel warp
with the same kernel:

| | frame-0 share now | with the reference resampled | ground truth |
|---|---|---|---|
| GPU | 53.6% | **5.3%** | 2.9% |
| CPU | 11.7% | **9.1%** | 2.9% |

That is the bias essentially gone on the GPU path and materially reduced on the
CPU one, for the cost of a single extra warp per render.

### Fixed in 1.30.4 - and the half pixel the simulation used is the wrong size

The three stages all end the same way: compute the common valid region, build an
integer crop translation `T_crop`, and warp each frame by `T_crop . H_i`. So the
place to put the missing resample is `T_crop`, which every frame shares. Giving
it a sub-pixel offset takes the reference off the pixel grid, and because the
offset is common to all frames it cannot move them relative to each other - the
same single `warpPerspective` per frame simply receives a matrix whose
translation is no longer integral. The three copies became one
`_build_crop_transform` helper, applied whether or not the crop is degenerate.

That leaves one number to choose, and the half pixel the simulation above used
is not it. Interpolation loss is worst at the half-pixel phase and zero on the
grid, so half a pixel does not equalise the anchor, it **over-corrects**:
measured on a stack whose frames carry identical content, so that the honest
answer is "no frame is sharper than any other",

| offset | anchor vs the frames it anchors, Lanczos4 | bilinear |
|---|---|---|
| 0 (the defect) | **+22.8%** | **+85.9%** |
| 0.15 px | +13.4% | +49.2% |
| **0.25 px** | **+1.1%** | **+3.4%** |
| 0.30 px | -7.6% | -20.2% |
| 0.40 px | -18.1% | -55.7% |
| 0.5 px (the simulation) | **-22.5%** | **-72.3%** |

Half a pixel leaves the anchor 22% softer than the stack on the Lanczos path and
72% softer on the bilinear one, which inverts the bias rather than removing it -
a frame that is never selected is as wrong as one that is always selected. A
quarter pixel is where the anchor's sharpness matches the stack mean on both
interpolators and wherever the anchor sits in the stack. It is close to the
0.211 phase at which bilinear attenuation equals its average over a uniformly
distributed sub-pixel offset, which is the treatment an arbitrary frame's
residual translation actually gets. Two independent criteria pick it: on the
real scenes the selection share falls steeply up to 0.25 px and then plateaus
near ground truth out to 0.5 px, so 0.25 is the near edge of that plateau rather
than a point on a slope, and the sharpness balance above is what selects 0.25
from within it.

**The measurement item 3 is stated in**, re-run before and after in one process
so the "before" column reproduces the tables above:

| scene | pipeline | CPU before | CPU after | GPU before | GPU after | truth |
|---|---|---|---|---|---|---|
| handheld_drift | scale | 11.7% | **4.3%** | 53.6% | **3.2%** | 2.9% |
| handheld_drift | ecc | 11.4% | **4.3%** | 51.4% | **3.3%** | 2.9% |
| handheld_drift | **scale+hom** (default) | 22.3% | **4.7%** | 61.2% | **3.4%** | 2.9% |
| handheld_drift | scale+hom+ecc | 33.8% | **5.1%** | 68.0% | **3.6%** | 2.9% |
| flower01_handheld | scale | 14.7% | **3.0%** | 57.2% | **0.1%** | 0.0% |
| flower01_handheld | **scale+hom** (default) | 29.2% | **3.3%** | 68.3% | **0.2%** | 0.0% |
| flower01_handheld | scale+hom+ecc | 43.3% | **4.5%** | 82.4% | **0.3%** | 0.0% |

Two thirds of the picture coming from the anchor frame becomes three percent of
it, against a truth of 2.9%. The compounding with pipeline length that this item
shared with item 7 goes with it: the CPU spread across one, two and three stages
was 11.7% -> 22.3% -> 33.8% and is now 4.3% -> 4.7% -> 5.1%.

**What it costs, stated plainly.** The anchor loses a free ride it should never
have had, so the fused picture is a little softer where it used to source that
frame. Against each scene's all-in-focus ground truth the change runs from
-0.70 dB to +0.18 dB, most rows landing between -0.24 and -0.14 dB; the worst
case is `scale+hom+ecc` on handheld_drift (30.79 -> 30.08 dB GPU, 30.70 -> 30.03
CPU), and `ecc` on flower01_handheld actually gains 0.11 dB on both devices.
That is the real price of the trade, and it is worth taking: the dB it gives up
is dB that was earned by sourcing the picture from a frame which is out of focus
there, and item 4 is where the sharpness properly comes back.

**Geometric accuracy is untouched**, which is the claim the shared offset has to
support. Every single-stage pipeline is bit-identical before and after on both
scenes and both devices - 2.22, 1.67, 2.57, 6.40 px to the last digit - as is
the kept-frame share, since the crop region is computed before the offset is
folded in. Only `scale+hom` moves, in both directions (handheld_drift 2.31 ->
2.23 px CPU and 2.68 -> 2.04 px GPU; flower01_handheld 1.50 -> 2.06 px CPU and
1.95 -> 2.32 px GPU), because it is the one pipeline whose second stage detects
SIFT features on pixels the first stage resampled, and the anchor's pixels have
now changed. This is item 7 showing through: composing the stages into a single
warp would remove the sensitivity along with the double interpolation.

**Guarded by** `tests/test_registration_reference_resample.py`, which fails in
both directions. Its sharpness assertion is two-sided on purpose - reverting to
an integer crop fails 9 of its 14 tests, and setting the offset to the half
pixel the simulation used fails 4 - so neither the defect nor its over-correction
can come back quietly. The frame-share half runs against the real scenes and
skips where `samples/` has not been generated.

---

## 4. GPU warping is bilinear where CPU warping is Lanczos4

**Category: quality. Impact: high. Effort: low. Fixed in 1.30.5.**

The CPU path warps with `cv2.INTER_LANCZOS4` (`core/registration.py:319`, `:674`,
`:973`). The GPU path uses `order=1` - bilinear - in both copies
(`core/registration.py:137`, `:956`). The stated reason is speed:

> Interpolation is bilinear (order=1) - map_coordinates has no Lanczos kernel,
> and the higher spline orders cost far more than the quality difference is worth
> on stacks.

The quality difference is not small, and the speed is not there. Sweeping the
spline order on `handheld_drift`:

| variant | time | mean sharpness | share picked from frame 0 |
|---|---|---|---|
| order=1 (bilinear, current) | 0.27s | 27.2 | 53.6% |
| order=3 (cubic spline) | 0.74s | 44.4 | 16.3% |
| order=5 (quintic) | 0.85s | 49.5 | 11.0% |
| **CPU `cv2.INTER_LANCZOS4`** | **0.27s** | **48.9** | **11.7%** |

The last column is a pre-1.30.4 reading and item 3 has since collapsed it to
3.2-4.3% for every row - it was measuring the two defects together. The
sharpness column is unaffected and is the one this item rests on.

Bilinear discards **44%** of the high-frequency energy Lanczos keeps
(27.2 vs 48.9), and the GPU warp that costs that is *not faster than the CPU
warp it replaces* at this size. On the largest sample stack, 14 frames at
2560x1430, the GPU advantage is 0.74s against 0.94s - 1.27x, for a stage that is
a fraction of any real render.

The same shows up against ground truth on a synthetic stack with a known
transform, where the Laplacian variance of each warped frame halves:

| | frame 1 | frame 2 |
|---|---|---|
| GPU warp (bilinear) | 619.4 | 554.9 |
| CPU warp (Lanczos4) | 1220.2 | 1142.5 |

For a focus-stacking application this is the wrong trade in the wrong place.
Every frame is resampled *before* the focus measure reads it, so half the
high-frequency detail the whole pipeline exists to find is destroyed on the way
in. Note the interaction with item 3: it is not merely that frames get softer, it
is that they get softer *unequally*, and the fusion reads that inequality as
depth.

**The fix has three options,** and the measurement points at the first:

- **Drop the GPU warp path.** It is not faster, it is materially worse, and it
  is the source of items 1, 3 (amplified), 4 and 10. Deleting it removes ~110
  lines and every GPU-versus-CPU divergence in this document.
- Raise it to `order=3`, which recovers most of the quality at 2.7x the GPU time
  (still ~0.74s, i.e. no worse than the CPU path it is replacing).
- Keep bilinear only where a preview is being generated and quality is not the
  point.

### Fixed in 1.30.5, by the first route - and the GPU is not slower, it is doing less

The second option was checked before the first was taken, because "raise the
spline order" keeps the acceleration and the accuracy both. It does not survive
its own timing. Sweeping the order on one 2560x1430 frame under a transform of
the kind the stages actually produce - a fraction of a pixel of translation plus
the magnification a stack breathes by - warmed up, so the numbers are
steady-state rather than CUDA JIT:

| warp | time | sharpness | vs Lanczos4 |
|---|---|---|---|
| CPU `cv2.INTER_LANCZOS4` | **0.012s** | 12.74 | - |
| GPU `map_coordinates` order=1 (what shipped) | 0.007s | 9.21 | 72% |
| GPU order=3 | 0.016s | 12.17 | 96% |
| GPU order=5 | 0.025s | 13.10 | 103% |

The GPU warp was only ever faster because it was doing less. At the order that
matches Lanczos4 it is 1.3x to 2.1x *slower* than the CPU warp it exists to
replace, on a 4080 against a frame the CPU handles in twelve milliseconds - the
transfer and the dense coordinate grid cost more than the sampling saves. There
is no operating point at which the second path wins, so there is no reason to
keep a second path. `_warp_perspective_gpu` and `_init_gpu_warp` are gone; the
three copies of the warp call became one `_warp_frame`, and the ECC stage's
serial-because-of-PCIe loop became the same thread pool the other stages use.

**On the machine this document is measured on** (RTX 4080, so previously the
bilinear path), fusing with Pyramid against each scene's all-in-focus ground
truth. "sharpness" is the Laplacian variance of the fused picture:

| scene | pipeline | sharpness before | after | PSNR before | after |
|---|---|---|---|---|---|
| handheld_drift | scale | 114.7 | **204.1** (+78%) | 30.04 | 29.91 |
| handheld_drift | ecc | 102.2 | **194.5** (+90%) | 30.70 | 30.51 |
| handheld_drift | **scale+hom** (default) | 103.5 | **185.8** (+80%) | 30.05 | 29.52 |
| handheld_drift | scale+ecc | 59.6 | **183.6** (+208%) | 30.22 | 29.74 |
| handheld_drift | scale+hom+ecc | 54.1 | **172.6** (+219%) | 30.08 | 30.03 |
| flower01_handheld | scale | 29.0 | **41.6** (+43%) | 29.16 | 29.06 |
| flower01_handheld | ecc | 29.3 | **41.7** (+42%) | 28.05 | 27.97 |
| flower01_handheld | **scale+hom** (default) | 26.9 | **39.8** (+48%) | 28.67 | 28.98 |
| flower01_handheld | scale+ecc | 18.2 | **39.8** (+119%) | 28.11 | 27.97 |
| flower01_handheld | scale+hom+ecc | 17.6 | **38.1** (+117%) | 27.97 | 27.84 |

Against the unregistered stack's own 214.1 and 42.6, the fixed pipeline now
gives up 5% of the picture's high-frequency energy to a single-stage
registration where it used to give up 46%, and the three-stage pipeline gives up
19% where it used to give up 75%. The compounding is what the last rows show:
each stage was another bilinear pass, so the loss multiplied with pipeline
length, and the pipelines the rest of this document recommends were the ones
paying it three times.

**The PSNR column moves the other way, by a tenth of a dB, and that is the
honest reading of it.** Ten of the twelve rows lose between 0.05 and 0.53 dB.
PSNR is a mean-squared-error score against a *rendered* reference, and the
residual on these scenes is dominated by sensor-like noise and by the sub-pixel
misregistration items 2 and 6 are about - both of which a narrow kernel
suppresses. Blur is therefore worth a fraction of a dB to it, which is the same
weakness [ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md) records for the
fusion audit and the reason this item is stated in high-frequency energy
instead. A metric that rewards a warp for discarding 44% of the detail in a
*focus-stacking* pipeline is measuring the wrong thing: the fusion stage decides
which frame to source each pixel from by asking which is locally sharpest, and
that question cannot be answered from detail the warp already threw away.

**What it costs in time**, whole-stage, this machine, best of two runs:

| stack | stage | GPU bilinear before | after (CPU Lanczos4) |
|---|---|---|---|
| flower01_subject_hires, 14x 2560x1430 | scale | 0.40s | 0.77s |
| flower01_subject_hires | ecc | 0.62s | 0.99s |
| handheld_drift, 16x 640x480 | scale | 0.20s | 0.20s |
| flower01_handheld, 14x 1280x715 | scale+hom+ecc | 1.26s | 1.42s |

A third of a second on the largest sample stack, and nothing measurable on the
small ones - where deleting the path is in fact *faster*, because initialising a
CUDA context cost more than the warps did (`scale` on handheld_drift ran 0.56s
with the GPU probe and 0.17s without it). Set against 44% of the detail the
whole application exists to find, on a stage that is a fraction of any real
render.

**Three other items resolve with it.** Item 10 - the 21x memory blowup with no
fallback, which would have failed outright on an 8 GB card at 60 MP - is gone
with the dense coordinate grid that caused it. The second half of item 9, the
ECC GPU branch that iterated `img.shape[2]`, went with item 1 and its last
traces go here. And every "GPU" row in this document is now the CPU row: the
`scale+hom` divergence item 1 left behind (2.04 px against 2.23 px on
handheld_drift, 2.32 against 2.06 on flower01_handheld) closes to a single
number, because it only ever existed because the second stage was detecting SIFT
features on bilinearly resampled pixels.

**Geometric accuracy is unchanged where it was already device-independent** -
every single-stage row is bit-identical to its old CPU reading - and the
selection shares item 3 reports converge on the CPU figures too (handheld_drift
`scale` 3.2% -> 4.3%, `scale+hom+ecc` 3.6% -> 5.1%, against a truth of 2.9%).
Those go slightly *up*, and for the same reason the sharpness did: a bilinear
warp softens every non-reference frame so heavily that the anchor wins fewer
pixels than it should, which flattered the number without deserving it.

**Guarded by** `tests/test_registration_warp_kernel.py`, which is
`test_registration_gpu_warp.py` retargeted at what replaced the path it used to
compare. It brackets each assertion between a bilinear and a Lanczos4 warp of
the same frame under the same matrix, so it asserts a kernel rather than a
constant: lowering `WARP_INTERPOLATION` to `INTER_LINEAR` fails 5 of its 10
tests, and reintroducing a GPU warp fails the two that record every import a
stage makes while registering. Item 1's drift assertion is kept as it was - a
warp that applies its transform backwards is not a device-specific mistake.

`--devices` is gone from `tests/benchmark_registration.py` along with the device
column in all three of its tables. There is one warp path now, so there is
nothing left for it to run twice.

---

## 5. `downscale_width` is silently overridden for every real camera file

**Category: robustness. Impact: medium. Effort: trivial. Fixed in 1.30.6.**

All three stages carry the same block (`core/registration.py:205-208`, `:541-548`,
`:732-738`):

```python
max_dim = max(h_orig, w_orig)
if max_dim >= 2048:
    downscale_width = 1024
```

That is an unconditional assignment, not a cap. A user who raises the setting to
2048 for a difficult stack gets 1024; a user who lowers it to 512 for speed also
gets 1024. Since the threshold is 2048 and the trigger is the *longer* side, it
fires on every frame from any camera made this century.

Measured, four settings against the same stack:

| scene | setting | time | output | result |
|---|---|---|---|---|
| flower01_handheld (1280x715) | 512 | 0.07s | 1259x696 | used as set |
| | 1024 | 0.12s | 1267x702 | used as set |
| | 2048 | 0.16s | 1257x694 | used as set |
| flower01_subject_hires (2560x1430) | 512 | 0.24s | 2555x1422 | **overridden -> 1024** |
| | 1024 | 0.24s | 2555x1422 | **overridden -> 1024** |
| | 2048 | 0.25s | 2555x1422 | **overridden -> 1024** |
| | 4096 | 0.30s | 2555x1422 | **overridden -> 1024** |

Byte-identical results across a 8x range of the setting. The control in the
settings dialog, the `reg_downscale_width` key in `openfocus.cfg.json`, and the
`downscale_width` argument threaded through `ImageRegistration`, `RenderWorker`
and `BatchWorker` are all inert on real input.

This also sets a hard accuracy floor nobody chose. Detection at 1024 on a
6000 px frame means a keypoint located to ±0.5 px is known to ±2.9 px at full
resolution, which is the same order as the residuals in item 2 - so some of what
looks like estimator error is quantisation.

**The fix is to make it a cap rather than an assignment** -
`downscale_width = min(downscale_width, 1024)` preserves today's behaviour for
anyone who has not touched the setting while letting the setting mean something -
or, better, to remove the override and let the value stand, since it exists
precisely so the user can make this trade. Either way the three copies should
become one helper.

### Fixed in 1.30.6, by the second route - the cap only half-fixes it

The override is gone rather than capped, and the three copies are one
`_resolve_detection_width`. The cap was the safer-sounding option and it is the
wrong one: it leaves a user who raises the setting for a difficult stack in
exactly the position this item describes, silently getting 1024 while believing
they asked for more, and the measurements below are precisely the accuracy that
half of the range was hiding. Nothing is lost by removing it, because
`REG_DOWNSCALE_WIDTH`, `openfocus.cfg.json` and `ImageRegistration` all default
to 1024 already - a user who has not touched the setting gets the same detection
resolution as before, and now gets it because it is the default rather than
because the value was overwritten.

The three stage defaults were inconsistent behind the override - 1600 for
`scale` and `homography`, 1000 for `ECC` - and are now the one
`DEFAULT_DETECTION_WIDTH = 1024` the app ships. That is only reachable from
library and CLI use; every in-app caller passes the setting explicitly.

**What the setting is worth**, on a stack large enough that the override used to
fire. Neither scene carrying a per-frame affine is that big - 640 and 1280 px -
so the measurement runs on `flower01_subject_hires` (2560x1430) put through the
same breathing/drift affine `samples/generate_samples.py` applies to
`handheld_drift`, which makes the ground truth exact at full size. Every row
below was the same number before the fix, because every row was detected at
1024:

| pipeline | 512 | 1024 (what you used to get) | 2048 | 2560 = full resolution |
|---|---|---|---|---|
| scale | 3.65 px | 6.77 px | 5.47 px | **4.80 px** |
| homography | 3.65 px | 6.77 px | 5.47 px | **4.04 px** |
| ecc | 12.54 px | 12.98 px | 11.49 px | **10.34 px** |
| both (hom+ecc) | 13.06 px | 13.34 px | 11.26 px | **9.53 px** |
| scale+hom | 4.48 px | 5.05 px | 3.19 px | **2.89 px** |
| scale+ecc | 13.06 px | 13.34 px | 11.26 px | **9.50 px** |
| scale+hom+ecc | 13.17 px | 13.63 px | 10.85 px | **9.11 px** |

Full-resolution detection beats the forced 1024 on all seven pipelines, by
1.26x to 1.75x, and a second noise seed (`--seed 313`) reproduces that on all
seven. **The intermediate columns do not order themselves**, and the 512 column
is the honest illustration of why: it beats 1024 on this seed and loses to it on
the other, because at that reduction SIFT is finding a different, smaller set of
keypoints rather than the same ones less precisely. So this is not a "more is
better" dial below full resolution - what the measurement supports is that 1024
is not the best value on a full-size frame, not that the error falls smoothly as
the width rises.

Above the frame width nothing changes: `_detection_scale` never upsamples, so
4096 and 2560 are the same run on a 2560 px frame, to the last digit.

**What it costs.** Full-resolution detection is roughly 2x the stage time for
the feature-based stages and 3.4x for ECC, on 14 frames of 2560x1430:

| pipeline | 1024 | full resolution |
|---|---|---|
| scale | 1.10s | 2.76s |
| homography | 1.46s | 2.62s |
| ecc | 1.65s | 5.55s |
| scale+hom+ecc | 3.97s | 10.09s |

That is the trade the setting exists to offer, and it is now the user's to make.
The default does not move: 1024 stays the shipped value, so nobody pays this
without asking for it.

**The table this item opens with, re-run.** Running `scale` on
`samples/flower01_subject_hires` at each setting used to give 2555x1422 four
times over, byte-identical across an 8x range; it now gives 2543x1405,
2550x1414, 2555x1419 and 2551x1422 for 512, 1024, 2048 and 4096 - four different
alignments, which is what a setting that does something looks like.

**Guarded by** `tests/test_registration_downscale_width.py`. It records the
width each stage actually resamples to before SIFT or ECC sees it, on a frame
over the old 2048 threshold, and asserts it is the width the caller asked for -
including a width large enough that no downscaling happens at all, which the
override made unreachable. Reinstating the override fails 20 of its 39 tests,
among them the accuracy assertion that full-resolution detection is materially
better than detection at 1024 on a large frame, which is the whole of what the
setting buys.

---

## 6. The reference frame defaults to `first`, and `middle` is better everywhere

**Category: quality. Impact: medium. Effort: trivial.**

`resolve_reference_index` supports `first`, `middle` and `last`, and
`_rereference_transforms` implements the re-anchoring exactly - both are well
built and well tested. The default is `first`, which is the historical behaviour
rather than the good one.

Anchoring on the middle frame halves the longest chain, so accumulated error
spreads symmetrically instead of piling up at one end. It won on every pipeline
and both scenes measured, and it keeps more of the frame as a side effect,
because the union of displacements to be cropped away is smaller:

| scene | pipeline | first: mean px / kept | middle: mean px / kept |
|---|---|---|---|
| handheld_drift | scale | 2.22 / 87.3% | **2.15 / 91.2%** |
| handheld_drift | ecc | 1.67 / 88.8% | **1.55 / 92.3%** |
| handheld_drift | homography | 5.00 / 81.1% | **3.76 / 88.9%** |
| flower01_handheld | scale | 2.57 / 93.2% | **1.73 / 96.0%** |
| flower01_handheld | ecc | 6.40 / 91.8% | **2.93 / 94.8%** |
| flower01_handheld | scale+hom | 3.37 / 88.3% | **1.86 / 93.8%** |

The largest single improvement in this document that costs one changed default:
`scale` on flower01_handheld goes from 2.57 px to 1.73 px and keeps 3% more of
the frame. focus-stack already defaults to the middle frame
([ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md) §2).

The reason to hesitate is reproducibility - existing projects would re-render
differently. That is the same argument that kept `scale` opt-in, and it is worth
weighing against a free 30% accuracy gain.

---

## 7. Each stage is its own resample and its own crop

**Category: quality. Impact: medium. Effort: medium.**

`ImageRegistration.process` in `both` mode runs homography to completion -
estimate, warp, crop - and feeds the *pixels* to ECC, which estimates, warps and
crops again (`core/registration.py:1236-1245`). `RenderWorker._run_registration`
stacks `scale` in front of that the same way. Two consequences, both measured.

**Interpolation is paid per stage.** Each warp is a fresh resample of an already
resampled image, and resampling is not idempotent. Pushing one real frame
through repeated half-pixel warps - the sub-pixel correction each stage
typically applies - and reading the high-frequency energy that survives:

| kernel | start | after 1 warp | after 2 | after 3 |
|---|---|---|---|---|
| `INTER_LANCZOS4` (all three stages, as of 1.30.5) | 30.0 | 23.9 (79.7%) | 23.8 (79.4%) | 20.7 (68.9%) |
| bilinear (the GPU path, deleted in 1.30.5) | 30.0 | 9.3 (31.2%) | 6.8 (22.7%) | 4.7 (15.7%) |

Before 1.30.4 the reference frame was exempt (item 3), so that loss landed on
every frame *except* one and the artificial advantage compounded with pipeline
length. On flower01_handheld, CPU path, comparing the reference frame against
the mean of the rest:

| pipeline | reference | others | reference advantage |
|---|---|---|---|
| unregistered | 26.3 | 30.0 | **-12.2%** (the reference starts *behind*) |
| scale | 26.3 | 23.9 | +10.3% |
| scale+hom | 26.3 | 20.5 | +28.8% |
| scale+hom+ecc | 26.5 | 17.9 | **+47.8%** |

The reference frame begins 12% *less* sharp than the average frame and ends 48%
sharper than it, without a single photon changing. Three stages is enough to
invert the picture's own focus ordering.

Item 3 removed the *inequality* here - the reference is now resampled with the
rest at every stage - and item 4 removed the second row, since every machine now
takes the first. Neither removes the loss itself: the first row is still paid,
once per stage, by every frame including the anchor, which is why a three-stage
pipeline gives up 19% of the picture's high-frequency energy where one stage
gives up 5%. That is this item, and it is the only one of the four resampling
defects still open.

**Pixels are paid per stage too.** Every crop takes the intersection of the valid
regions, and the intersections compose:

| scene | scale | homography | both | scale+hom+ecc |
|---|---|---|---|---|
| handheld_drift | 87.3% | 81.1% | 76.1% | **77.7%** |
| flower01_handheld | 93.2% | 91.2% | 87.5% | **83.8%** |

So the full pipeline discards a fifth of the frame on handheld_drift and a sixth
on flower01_handheld, having resampled the survivors three times, to arrive at an
alignment that item 2 shows is no better than `scale` alone.

**The fix is to compose the transforms and warp once.** Each stage already
produces a 3x3 matrix; estimating stage 2 on stage 1's *transformed coordinates*
rather than on its output pixels, then applying `T_crop · H₂ · H₁` in a single
`warpPerspective`, gives one interpolation and one crop for any number of stages.
The stages already agree on the convention, so the composition is a matrix
product. This also subsumes half of item 3: with one warp, there is only one
frame to keep honest.

---

## 8. Transforms are accepted on six matches, with the inlier mask discarded

**Category: quality. Impact: medium. Effort: low.**

Both feature-based stages gate on the number of matches surviving the ratio test
and nothing else (`core/registration.py:261`, `:615`):

```python
if len(good_matches) < 6:
    print(f"Warning: Frame {idx} poor matches ({len(good_matches)}).")
```

Six is two above the four a homography needs. RANSAC's own verdict on the fit is
then thrown away - `mask` at line 625 is assigned and never read again - and the
resulting matrix is accepted with no check on its determinant, its perspective
terms or its scale.

The pair that sits on the floor is exactly the pair that goes wrong. On
handheld_drift:

| pair | good matches | homography error | keystone introduced |
|---|---|---|---|
| 13 | 15 | 1.271 px | 2.00% |
| 14 | 13 | 1.077 px | 0.95% |
| **15** | **6** | **4.465 px** | **5.67%** |

The 6-match pair is 3.5x worse than its neighbours and bends the frame by 5.7%.
Because the chain accumulates - `H_global = np.matmul(H_global, H_local)` - a
bad estimate is not a bad frame, it is a permanent offset applied to every frame
after it.

What the code does well and should be credited for: when a pair *is* rejected,
`last_kps`/`last_des` are deliberately not advanced, so the next frame is matched
against the last frame that worked and the chain re-closes correctly. Verified by
blanking frame 4 of an 8-frame stack - frames 5 through 7 come back at 0.11-0.13
px residual, entirely unharmed. The recovery machinery is right; only the
decision about *when* to invoke it is too permissive.

**The fix is to gate on the RANSAC verdict rather than the match count** - a
minimum inlier count and inlier ratio, plus a sanity check on the estimated
matrix (perspective terms near zero, scale within a few percent of 1, positive
determinant) - and to fall back to the existing skip path when it fails. The
skip path already works.

**Partly mitigated by item 2 (1.30.3), not resolved.** The homography stage now
reads the RANSAC masks it used to discard, and a pair with fewer than 24 inliers
is fitted with the 4-DOF similarity rather than the 8-DOF homography - so the
6-match pair above no longer gets to bend the frame by 5.7%. It is still
accepted, still on nothing but a match count, and `scale` still discards its
mask entirely.

---

## 9. ECC crashes on single-channel input

**Category: robustness. Impact: low. Effort: trivial.**

`_align_ecc_impl`'s preprocessing calls `cv2.cvtColor(small_img,
cv2.COLOR_BGR2GRAY)` unconditionally (`core/registration.py:763`), and its GPU
branch iterated `for c in range(img.shape[2])` (`:950`). Both assume three
channels. On a grayscale stack:

| stage | result |
|---|---|
| scale | OK -> (502, 500) |
| homography | OK -> (501, 499) |
| **ecc** | **`cv2.error: Bad number of channels`** |

The second half went with item 1's fix, which routed the ECC stage through the
shared GPU helper rather than its own channel loop, and the helper itself went
with item 4 - `cv2.warpPerspective` has never cared how many channels it is
handed. The first half is still open: a two-line guard on the colour conversion.

Low impact because the app's own loader normalises to BGR, so this is reachable
from the CLI entry point and from library use rather than from the GUI.

---

## 10. The GPU warp helper allocates 21x the frame, with no fallback

**Category: performance. Impact: low. Effort: low. Removed with the warp in 1.30.5.**

Both GPU warps build a full dense coordinate grid before sampling: an int64
`meshgrid`, a 3xN stack of homogeneous coordinates, the float64 matrix product,
and two float64 coordinate planes. That is a fixed multiple of the frame,
measured against the CuPy pool:

| frame | size on host | GPU pool peak | ratio |
|---|---|---|---|
| 6 MP uint16 BGR | 36.0 MB | 767.7 MB | 21.3x |
| 12 MP | 72.0 MB | 1536.0 MB | 21.3x |
| 24 MP | 143.9 MB | 3070.9 MB | 21.3x |
| 60 MP | 359.9 MB | 7678.2 MB | 21.3x |

A 60 MP frame completed here on a 16 GB card. On an 8 GB card it would not, and
neither `_warp_perspective_gpu` nor either `warp_task` has a `try`/`except`
around the CuPy calls - so the allocation failure would propagate out of the
worker thread rather than falling back to the CPU path that is sitting right
there. This is headroom rather than an observed failure, and it is stated as
such.

The grid does not need to be dense or float64: the coordinates could be built in
float32 and streamed in row blocks, or the whole thing replaced by CuPy's own
affine machinery for the affine case. But given items 1 and 4, the simplest
resolution is again to delete the GPU warp rather than optimise it.

**Resolved in 1.30.5**, by that route. Item 4 deleted the warp, and the dense
coordinate grid went with it, so there is no longer an allocation to bound or a
fallback to be missing. This item was headroom rather than an observed failure
and it was never fixed on its own terms - it stopped existing.

---

## 11. `_stabilisation_impl` is unreachable code

**Category: maintenance. Impact: low. Effort: trivial.**

117 lines (`core/registration.py:1006-1122`) implementing a Lucas-Kanade
trajectory-smoothing stabiliser. It is not in `SUPPORTED_METHODS`, not dispatched
from `process`, and its only two references in the repository are inside
commented-out blocks (`:1151`, `:1371`). It also carries a hardcoded 1.04
zoom-crop and would raise `IndexError` on a single-frame input at
`transforms[-1] = transforms[-2]`.

Trajectory smoothing is a video idea rather than a stacking one - it deliberately
*keeps* low-frequency motion, which is exactly what a stack wants removed - so
this is a deletion, not a revival. The same applies to the commented-out
`_registration_impl` and the four commented-out compatibility aliases below it,
which reference an `_align_zoom_impl` that no longer exists.

---

## 12. Three copy-pasted folder loaders that disagree

**Category: robustness. Impact: low. Effort: low.**

The same ~12-line directory loader appears in `_align_scale_impl` (`:183-193`),
`_align_homography_impl` (`:516-529`) and `_align_ecc_impl` (`:712-721`), plus a
fourth variant in the dead `_stabilisation_impl`. They have already drifted:

```python
# scale and homography
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# ecc
{'.jpg', '.jpeg', '.png', '.bmp', '.tif'}          # .tiff missing
```

So a folder of `.tiff` files aligns under `scale` and `homography` and silently
loses every frame under `ecc`. The dead fourth copy sorts with
`int(re.findall(r"\d+", ...)[-1])` and no fallback, so it raises `IndexError` on
any filename without a digit - the same defect
[ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md) §13 records for the fusion
loaders, and the fix should be shared with it: one loader, one extension list,
one sort with a fallback.

Low impact because every in-app caller passes a preloaded list
(`core/workers.py:96`, `:366`, `:381`, `:649`); only `main()` and library users
reach these paths.

---

## What already works

Several things were probed and found sound; they are recorded so the audit is
not read as uniformly negative.

- **The chain's failure recovery is correct.** Rejecting a pair without advancing
  the feature reference means the next frame matches against the last good one
  and the chain closes properly. A blanked frame in the middle of a stack leaves
  every subsequent frame at 0.11-0.13 px (item 8).
- **`_rereference_transforms` is exact** - the reference collapses to identity,
  every other transform is rewritten relative to it, index 0 is a true no-op, and
  a singular reference falls back rather than raising.
- **`ecc_parallel` is genuinely free.** Concurrent and serial pair computation
  produce bit-identical output (max abs diff 0), because the pairs are
  independent and only the accumulation is ordered.
- **Registration is deterministic.** `both` run twice on the same input gives max
  abs diff 0.
- **The ECC homography rescale is right.** Converting a homography measured at
  scale s to full resolution needs `S⁻¹HS`, which is exactly the four assignments
  at `core/registration.py:818-822`.
- **Bit depth is preserved.** 16-bit input comes out 16-bit with no clipping at
  either end on every stage and both devices - the one exception being the GPU
  ECC zeros, which are item 1 and not a depth problem.
- **ECC's CPU path is the most accurate estimator in the codebase**, landing
  within 0.02 px on a clean translation stack. Item 1 is the only reason it is
  not the recommended stage.

---

## What to run today

There is no longer a "which machine" to this - as of 1.30.5 registration warps
the same way everywhere, so the advice below is the advice on every machine:

- **`ecc` is the most accurate stage** (2.36x on handheld_drift). Item 1 was the
  only reason to avoid it where CuPy was installed, and that reason is gone
  twice over: 1.30.2 fixed the sign error, and 1.30.5 removed the path.
- **Use `scale` alone** where ECC is too slow or the stack is mostly breathing.
  It is the most reliable single stage on both scenes (1.78x and 2.65x) and the
  only one that was never worse than doing nothing.
- **Set the reference frame to `middle`** (item 6). Free, better everywhere.
- **`homography` is safe to leave on** as of 1.30.3, and is worth turning on
  where the camera may have tilted between frames - it now falls back to the
  same constrained model `scale` fits wherever the perspective is not real
  (item 2). It is still not worth stacking on top of `scale` for its own sake.
- **The `downscale_width` setting works** as of 1.30.6, on real camera files as
  well as small ones (item 5). Leave it at 1024 unless a stack is misbehaving;
  raising it to the frame width is worth 1.26x to 1.75x of geometric accuracy on
  a 2560 px stack, for roughly twice the stage time - and lowering it now
  genuinely buys speed rather than being ignored.
- **Nothing needs doing about the anchor frame** as of 1.30.4: every frame is
  now resampled once, so the focus measure no longer prefers whichever frame the
  alignment happened to be anchored on (item 3).
- **Nothing needs doing about the GPU** as of 1.30.5: there is no GPU warp to
  avoid, and `ecc` or `scale`+`ecc` is the best-looking pipeline as well as the
  best-aligned one, wherever it runs.

What is left to watch is pipeline length. Every stage is still its own resample
(item 7), so `scale`+`hom`+`ecc` gives up 19% of the picture's high-frequency
energy where one stage gives up 5%, and crops a fifth of the frame to arrive at
an alignment barely better than `ecc` alone.

---

## Reproducing these measurements

Everything above comes from `tests/benchmark_registration.py`, which runs the
real `ImageRegistration` and reads its transforms back out by wrapping the crop
helper every stage calls - so the numbers describe the shipping code rather than
a reimplementation of it.

```bash
# geometric error of every pipeline against per_frame_affine ground truth
python tests/benchmark_registration.py

# fused PSNR/SSIM and sharpness against each scene's all_in_focus.png (items 1, 4)
python tests/benchmark_registration.py --quality

# which frame the focus measure picks, against focus_index.png (item 3)
python tests/benchmark_registration.py --selection

# the reference-frame comparison (item 6)
python tests/benchmark_registration.py --reference middle

# geometric error against the detection width, full-size stack (item 5)
python tests/benchmark_registration.py --detection-width
python tests/benchmark_registration.py --detection-width --seed 313 --widths 1024,2560

# one scene, one setting
python tests/benchmark_registration.py --scene flower01_handheld --downscale-width 2048
```

The scenes are `samples/handheld_drift` and `samples/flower01_handheld`,
regenerated deterministically by `python samples/generate_samples.py`. They are
the only two that carry `per_frame_affine`, which is what makes the geometric
error exact; `--scene` on any other name reports that and skips.

`--detection-width` is the exception, because both of those scenes are smaller
than the 2048 px the override triggered on. It builds its own stack instead -
the frames of `samples/flower01_subject_hires` (2560x1430) put through the same
`breathing_transform` the sample generator applies to `handheld_drift`, so the
ground truth is exact at a size a camera actually produces. `--seed` changes the
noise, which is how the ordering of the intermediate widths was found to be
unstable while the full-resolution column was not.

Everything runs once now. The `--devices` flag and the device column in all
three tables went with item 4: there is one warp path, so there is no second
answer to print.

The existing contract tests remain the guard against regression:

```bash
python -m pytest tests/test_registration_scale.py tests/test_registration_reference.py \
                tests/test_registration_warp_kernel.py \
                tests/test_registration_homography_model.py \
                tests/test_registration_reference_resample.py \
                tests/test_registration_downscale_width.py -v
```

The first two would not catch item 4 - `test_registration_scale.py` asserts
convergence thresholds loose enough to pass on either interpolator. The third
arrived with item 1's fix, as `test_registration_gpu_warp.py`, and was retargeted
by item 4's at what replaced the path it used to compare against. It brackets
each assertion between a bilinear and a Lanczos4 warp of the same frame under
the same matrix, so it holds the *kernel* rather than a constant: lowering
`WARP_INTERPOLATION` fails 5 of its 10 tests, and reintroducing a GPU warp fails
the two that record every import a stage makes while registering. It no longer
skips anywhere - the invariant it guards is the same on a machine with a CUDA
device and one without, which is the point of the change it came from.

The fourth arrived with item 2's fix and measures the same geometric error this
document does, reading the stage's transforms back through the same crop-helper
spy: it asserts that the stage beats leaving the stack alone on both
ground-truth scenes (it does not, on the pre-1.30.3 estimator) and that a stack
carrying a real camera tilt still comes back sub-pixel (it does not, if the
stage is changed to always fit a similarity). The scene half skips where
`samples/` has not been generated.

The fifth arrived with item 3's fix and is the one guard here that fails in
*both* directions by construction. It registers a stack whose frames carry
identical content, so the honest answer is that no frame is sharper than any
other, and asserts two-sidedly that the anchor comes out neither sharper nor
softer than the frames it anchors: restoring the integer crop fails 9 of its 14
tests, and moving the offset to half a pixel fails 4. It also checks the frame
share on the real scenes against `focus_index.png`, skipping where `samples/`
has not been generated, and that registering with and without the offset crops
identically and aligns equally well - which is what makes the shared offset a
resample rather than a misalignment.

The sixth arrived with item 5's fix and is the one guard here that needs a frame
larger than any sample scene: it builds a 2176 px stack, over the threshold the
override fired on, and records the width each stage actually resamples to before
SIFT or ECC reads it. That number has to be the width the caller asked for, at
every setting including one large enough to skip the downscale entirely.
Reinstating the override fails 20 of its 39 tests - the recorded widths, the two
settings that must now give two different results, and the accuracy assertion
that full-resolution detection beats detection at 1024 on a large frame. It
needs no samples and skips nowhere.
