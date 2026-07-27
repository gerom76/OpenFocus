# Focus stacking algorithms: OpenFocus against the field

A capability comparison between OpenFocus 1.8.0, the improvements proposed in
[ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md) and this document, and five
other focus stacking tools.

**Evidence quality differs by column and is marked as such.** OpenFocus, focus-stack
and Shine Stacker are open source, so their entries come from reading code or
project documentation. Helicon Focus, Zerene Stacker and PICOLAY are closed
source; their entries come from vendor documentation and long-standing community
consensus, and describe *claimed* behaviour. Where a vendor is silent, the cell
says so rather than guessing.

| Tool | Licence | Basis for entries here |
|------|---------|------------------------|
| **OpenFocus** 1.8.0 | MIT, open | source inspection |
| **Helicon Focus** 8 | commercial, closed | vendor docs |
| **Zerene Stacker** | commercial, closed | vendor docs + community |
| **focus-stack** (P. Aimonen) | GPL, open | project README |
| **Shine Stacker** (L. Lista) | open | project docs |
| **PICOLAY** (H. Cypionka) | freeware, closed | vendor docs |

---

## 1. Blending method families

This is the clearest single result in the comparison.

| Family | OpenFocus | Helicon | Zerene | focus-stack | Shine Stacker | PICOLAY |
|--------|-----------|---------|--------|-------------|---------------|---------|
| **Laplacian / contrast pyramid** | ✓ Pyramid³ | Method C | PMax | — | `PyramidStack` | — |
| **Depth map, hard per-pixel select** | ✓ Depth Map (Max)¹ | Method B | DMap | — | `DepthMapStack` (`DM_MAP_MAX`) | core method |
| **Contrast-weighted average** | ✓ Depth Map (Average) | Method A | — | — | `DepthMapStack` (`DM_MAP_AVERAGE`) | — |
| **Complex wavelet** | DTCWT² | — | — | ✓ (Forster et al. 2004) | — | — |
| **Block DCT** | ✓ | — | — | — | — | — |
| **Single-scale guided filter** | GFF, GFG-FGF | — | — | — | — | — |
| **Neural** | StackMFF V4, IFCNN | — | — | — | — | — |

¹ Added in 1.8.0 as `Depth Map (Max)` (`fusion_methods/depthmap.py`), an
order-independent per-pixel argmax on a pooled Laplacian focus measure. The other
methods still compute internal decision maps that are discarded
(`fusion_methods/gff.py:133`, `gfg_fgf.py:241`, `dct.py:107`), and no method yet
*exports* the depth map as a file — see §6.
² Applied recursively pairwise, so the result is order-dependent
([ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md) §7).
³ Added in 1.7.0. A single decision over all frames' Laplacian bands, so —
unlike DTCWT — it is order-independent (`fusion_methods/pyramid.py`). Since
1.19.0 that decision is a peak-relative weighting rather than a choose-max, for
the reasons in [ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md) §19; the
published rule remains selectable.

### What this table says

**As of 1.8.0 OpenFocus ships every blending family the field treats as
mandatory.** The Laplacian pyramid (`Pyramid`, 1.7.0) closed what this comparison
had flagged as the clearest gap, and the depth-map family followed in 1.8.0 as
`Depth Map`, carrying both selection rules the mature tools offer: a hard
per-pixel select (`Max`, cf. Zerene DMap and Helicon Method B) and the
contrast-weighted average (`Average`, cf. Helicon Method A and Shine Stacker's
average mode) that recovers multi-frame SNR in flat regions. Every commercial and
open competitor makes a pyramid or a closely-related multiscale transform a
headline method — Helicon's Method C, Zerene's PMax, Shine Stacker's
`PyramidStack`, and focus-stack's complex wavelet is the same idea in a different
basis — and pairs it with a depth map. OpenFocus now has both, plus DTCWT as a
second multiscale transform; unlike DTCWT, which fuses recursively pairwise and is
order-dependent, both the pyramid and the depth map decide across all frames at
once and are order-independent.

Conversely, **OpenFocus is the only tool with a block-DCT method and the only one
with neural fusion.** Those are genuine differentiators, and breadth is now well
clear of the field: nine fusion methods, where no competitor here offers more than
three.

The industry pattern is also worth noting: the mature tools converge on **exactly
two** methods, one pyramid and one depth-map, and teach users to retouch between
them. Helicon (A/B/C) and Zerene (PMax/DMap) both do this. Shine Stacker
replicates it. That pairing is deliberate — pyramid handles overlapping detail
and hair, depth-map keeps colour and noise clean, and neither dominates.

---

## 2. Geometric alignment

| Capability | OpenFocus | Helicon | Zerene | focus-stack | Shine Stacker | PICOLAY |
|------------|-----------|---------|--------|-------------|---------------|---------|
| Translation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Rotation | ✓ (via 8-DOF) | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Scale / focus breathing** | **✓** (similarity, 1.9.0) | ✓ magnification | ✓ default on | ✓ | ✓ | ✓ |
| Perspective / homography | ✓ | not documented | ✓ | ✓ | not documented | not documented |
| Feature-based (SIFT) | ✓ | not documented | not documented | — | not documented | not documented |
| Intensity-based (ECC) | ✓ | not documented | not documented | ✓ | not documented | not documented |
| **Reference frame** | **first/middle/last, selectable** (1.11.0) | not documented | not documented | **middle frame, selectable** | not documented | not documented |
| **Direct-to-reference option** | **✗** | n/a | n/a | **✓** | n/a | n/a |
| Non-rigid / local warp | ✗ | ✗ | ✗ (explicitly global-only) | ✗ | ✗ | ✗ |

### Findings

**Scale correction shipped in 1.9.0** as a dedicated `scale` registration method
(`_align_scale_impl` in `core/registration.py`), closing the one alignment gap
the whole field had already closed. Focus breathing is a 4-DOF similarity — a
uniform magnification about the optical axis plus a small rotation and recentring
— so the method fits exactly that with `cv2.estimateAffinePartial2D` under RANSAC
from SIFT matches, chained back to the first frame and cropped to the common
valid region like the other methods. Fitting a constrained similarity avoids the
overfitting an 8-DOF homography risks on frames that differ in blur. It is exposed
as its own "Scale (focus breathing)" toggle and composes ahead of Homography/ECC,
which then refine the residual translation and rotation. The earlier dead
`_align_zoom_impl` (a two-point SIFT scale ratio, linearly interpolated) is
removed, and the ECC docstring no longer over-claims focus-breathing suitability.
Unlike Zerene it is opt-in rather than default-on, to keep existing results
reproducible.

**The selectable reference frame shipped in 1.11.0**, closing most of the drift
gap. The alignment chain can now be anchored on the first (default), middle or
last frame; anchoring on the middle halves the longest chain, so accumulated
error spreads symmetrically instead of compounding toward one end — matching
focus-stack's default-middle behaviour. Mechanically the pairwise chain is still
accumulated against frame 0 (`H_global = np.matmul(H_global, H_local)` at
`core/registration.py:210`, `:549` and `:766`), then re-referenced onto the
chosen frame in one step (`_rereference_transforms`), which post-composes every
transform with the inverse of the reference frame's — the reference frame
collapses to identity and is held fixed, never warped.

**What remains is the direct-to-reference option.** focus-stack can also align
each frame *straight* to the reference instead of chaining neighbour-to-neighbour;
OpenFocus still builds its transforms by chaining. Re-anchoring the chain removes
most of the drift, but not the residual that accumulates along the chain itself —
skipping the chain entirely would remove that too.

**Nobody does non-rigid alignment.** Zerene's documentation states its alignment
is limited to whole-frame shift/rotate/scale. This is the one alignment axis
where an implementation would put OpenFocus ahead of the commercial tools rather
than level with them.

---

## 3. Photometric alignment

| Capability | OpenFocus | Helicon | Zerene | focus-stack | Shine Stacker | PICOLAY |
|------------|-----------|---------|--------|-------------|---------------|---------|
| **Brightness / exposure matching** | **✗** | ✓ | ✓ default on | ✓ default on | ✓ | not documented |
| White balance matching | ✗ | not documented | not documented | ✓ default on | ✓ (colour balance) | not documented |
| Contrast normalisation | ✗ | not documented | not documented | ✓ default on | ✓ | not documented |
| Vignetting correction | ✗ | not documented | not documented | not documented | not documented | not documented |

**Four of five competitors normalise brightness between frames before blending,
and three of those do it by default.** Zerene's docs attribute the need to flash
variation; focus-stack exposes it as `--no-whitebalance` / `--no-contrast`,
meaning both are on unless disabled.

OpenFocus does none of it. A grep for exposure, gain, or histogram matching
returns only unrelated hits in `controllers/label_manager.py:306`. The
consequence is not only visible luminance banding where the winning source frame
changes — it is that **every focus measure in the codebase is contrast-linear**
(Scharr in `gfg_fgf.py`, Laplacian in `gff.py:142`, DCT variance in `dct.py`,
squared-Laplacian band energy in `pyramid.py`, pooled Laplacian energy in
`depthmap.py`), so a 5% brighter frame wins the argmax on identical detail. This
is the single strongest catch-up signal in the whole comparison: cheap to
implement, near-universal in the field, and it corrupts the input to all nine
existing methods.

---

## 4. Focus measure and decision-map regularisation

| Capability | OpenFocus | Helicon | Zerene | focus-stack | Shine Stacker | PICOLAY |
|------------|-----------|---------|--------|-------------|---------------|---------|
| Analysis radius exposed | ✓ per method¹ | ✓ Radius | ✓ Estimation Radius | ✓ | ✓ `kernel_size` | ✓ |
| Smoothing radius exposed | partial² | ✓ Smoothing | ✓ Smoothing Radius | ✓ | ✓ | ✓ |
| Contrast / trust threshold | ✗ | ✗ | ✓ Contrast Threshold | ✓ background threshold | ✗ | ✓ noise suppression |
| Selectable focus operator | ✗ | ✗ | ✗ | ✗ | ✓ (Tenengrad, Laplacian, modified Laplacian, Sobel, variance) | ✗ |
| Local consistency filter | ✓ median / majority³ | implied | implied | ✓ `--consistency` 0-2 | ✓ | ✓ |
| **Global label optimisation (MRF / graph-cut)** | **✗** | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Noise-normalised focus measure** | **✗** | ✗ | ✗ | ✗ | ✗ | partial⁴ |

¹ `KERNEL_SIZE_DEFAULT_GFF = 31`, `_DCT = 7`, `_GFG = 7` in `constants.py`. Note
that GFF's exposed kernel was found to have little effect
([ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md) §9).
² Median kernel is exposed for DCT only; other methods use fixed smoothing.
³ `dct.py:133`, `dtcwt_torch.py:66`, guided-filter smoothing at `gfg_fgf.py:263`.
⁴ PICOLAY's noise-suppression parameter gates structure detection rather than
normalising the measure.

### Findings

**Zerene's Contrast Threshold has no OpenFocus equivalent and is the most
significant missing control.** DMap marks regions with too little contrast to
judge focus reliably, and stretches its depth estimate smoothly across them
instead of trusting noise. OpenFocus takes an unconditional argmax everywhere,
including in sky, defocused background, and smooth surfaces where the focus
measure is pure noise. The local median filter that follows can only make the
resulting speckle blockier.

**Global label optimisation is absent from every tool in the comparison.** All
six regularise the decision map with local filters. An MRF / graph-cut
formulation — data term from the focus measure, smoothness term penalising label
changes, down-weighted at image gradients — would be a differentiator, not a
catch-up.

**Shine Stacker is alone in exposing the focus operator itself.** Five
interchangeable energy measures is a cheap and genuinely useful feature; the
right operator is subject-dependent and no single choice wins everywhere.

---

## 5. Artifact handling and noise

| Capability | OpenFocus | Helicon | Zerene | focus-stack | Shine Stacker | PICOLAY |
|------------|-----------|---------|--------|-------------|---------------|---------|
| **Retouching brush from source frame** | **✗** | ✓ | ✓ | ✗ | ✗ | ✗ |
| Halo mitigation | ✓ dedicated radius (Depth Map)¹ | via Radius | via Radius | not documented | ✗ | ✗ |
| Denoising | ✗ | ✗ | ✗ | ✓ `--denoise` | ✓ non-local means | ✓ |
| Hot / noisy pixel masking | ✗ | ✓ dust map | ✗ | ✗ | ✓ automatic | ✗ |
| Sharpening | ✗ | ✗ | ✗ | ✗ | ✓ unsharp mask | ✓ |
| Multi-frame SNR gain in flat regions | ✓ Depth Map (Average) | ✓ Method A | ✗ | ✗ | ✓ average mode | ✗ |
| **Depth-wise slabbing / bunching** | **✗**² | ✗ | ✓ slabbing | ✓ `--batchsize` | ✓ `FocusStackBunch` | ✗ |

¹ Added in 1.14.0: the Depth Map methods grey-dilate each frame's focus energy
by a user-set halo radius before the per-pixel decision, so a sharply focused
edge claims the band its defocused glow contaminates in the other frames. This
is a dedicated control, not the indirect analysis-radius tuning Helicon and
Zerene document — see roadmap item 5.

² OpenFocus's tiling (`TILE_BLOCK_SIZE`, `TILE_OVERLAP`) subdivides in **X/Y for
memory**. Slabbing subdivides along the **depth axis for quality** — a different
operation with a different purpose.

### Findings

**Retouching is the commercial tools' answer to every artifact they cannot
solve algorithmically,** and Zerene's own documentation treats it as the expected
workflow: stack twice with PMax and DMap, then paint one into the other. Neither
open tool has it. Its absence caps how good any pure-algorithm result can be on
hard subjects.

**Depth-wise slabbing is a real gap discovered by this comparison.** Three of
five tools subdivide the stack along depth, stack each group, then stack the
results. It reduces artifact accumulation in deep stacks and is not what
OpenFocus's XY tiling does. This is a cheap addition given the existing worker
infrastructure.

**Helicon Method A and Shine Stacker's average mode recover multi-frame SNR, and
as of 1.8.0 so does OpenFocus.** Where all frames are equally defocused, N frames
offer a free √N noise reduction. Every other OpenFocus method hard-selects one
frame and throws that away; `Depth Map (Average)` instead blends by the focus
measure, so a flat region collapses to the plain mean and keeps the gain, while
detail still follows the sharpest frame.

---

## 6. Inputs and outputs

| Capability | OpenFocus | Helicon | Zerene | focus-stack | Shine Stacker | PICOLAY |
|------------|-----------|---------|--------|-------------|---------------|---------|
| **Bit depth** | ✓ 16-bit¹ | 16-bit | 16-bit | 16-bit TIFF | 16-bit | not documented |
| RAW support | Nikon only² | broad + DNG | via DNG | ✗ | not documented | ✗ |
| **Depth map output** | **✗**³ | ✓ (3D model) | ✓ | ✓ | ✗ | ✓ core feature |
| 3D / stereo / anaglyph output | ✗ | ✓ | ✓ | ✓ `--3dview` | ✗ | ✓ core feature |
| EXIF passthrough | ✓⁴ | ✓ | ✓ | not documented | not documented | not documented |
| ICC colour management | ✗ | ✓ | ✓ | not documented | not documented | not documented |
| Video / frame extraction input | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| GPU acceleration | ✓ CUDA/MPS | ✓ | ✗ | ✓ OpenCL | ✗ | ✗ |
| Batch / multi-folder | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Automatic stack ordering | ✗ filename only⁵ | order-sensitive (B) | ✓ auto order detect | ✗ | ✗ | ✗ |
| GUI | ✓ | ✓ | ✓ | ✗ CLI only | ✓ | ✓ |

¹ **Closed in 1.6.0.** Frames are decoded at native depth (`IMREAD_UNCHANGED`,
RAW at `output_bps=16`) and every stage is dtype-preserving, so a 16-bit stack
stays 16-bit through alignment, fusion and export. Mode selectable under
*Settings → Bit Depth*; see `utils/bitdepth.py`.
² `.nef` / `.nrw` at `core/image_loader.py:35`, though LibRaw handles CR2, CR3,
ARW, RAF, DNG and ORF identically.
³ Since 1.8.0 the map drives a fusion method (`Depth Map`) but is still not
*exported* as a file — see §1 note 1.
⁴ **Closed in 1.16.0.** A saved JPEG, PNG or JPEG XL inherits the EXIF block of
the first source frame of the render, and carries an XMP packet with OpenFocus'
version, render date, render duration and every option the render ran with
(`core/render_options.py`), plus a `Camera` section cloning every camera tag of
that source as text. Both are spliced into the encoded file, so nothing is
recompressed — see `utils/metadata.py`. TIFF and BMP outputs are still written
without metadata.
⁵ `core/image_loader.py:417`.

### Findings

**The 8-bit pipeline was the hard ceiling on everything above it — closed in
1.6.0.** Every other tool that documents bit depth works in 16-bit. Discarding
RAW's 12-14 bits before fusion meant banding in smooth gradients and no headroom
for weighted blending; it was not an algorithm, but it bounded every algorithm.
The pipeline now decodes at native depth and preserves it end to end, with the
depth mode selectable rather than implicit.

**Depth map output is standard, and OpenFocus computes one but still does not
export it.** Four of five competitors write it out; for PICOLAY it is the point of
the software. As of 1.8.0 the `Depth Map` method turns that internal map into a
blending method — closing depth-driven blending, one of the things this map is the
substrate for — and 1.14.0 added occlusion-aware halo suppression on the same
methods (a dedicated radius, ahead of the field's indirect tuning) — but saving
it as a file, and the 3D output it also feeds, are still open.

**OpenFocus leads on GPU and is alone in accepting video input.** CUDA/MPS
acceleration across five methods is ahead of everything except focus-stack's
OpenCL, and Zerene has no GPU support at all.

---

## 7. Proposed work, classified

Every item from the gap analysis, tagged by whether it closes a gap the rest of
the field has already closed, or moves ahead of it.

| # | Proposal | Class | Field precedent |
|---|----------|-------|-----------------|
| 1 | Photometric alignment (exposure/WB) | **Catch-up** | 4 of 5 tools; default-on in 3 |
| 2 | Scale / focus-breathing correction — **done in 1.9.0** | **Catch-up** | 5 of 5 tools |
| 3 | Laplacian pyramid fusion — **done in 1.7.0** | **Catch-up** | 3 of 5 directly, 4th equivalent |
| 4 | Global label optimisation (graph-cut) | **Differentiating** | none |
| 5 | Halo / bleed suppression — **done in 1.14.0** (Depth Map halo radius) | **Parity+** | only indirect, via radius tuning |
| 6 | Selectable reference frame — **done in 1.11.0** (direct-to-reference still open) | **Catch-up** | focus-stack ships both |
| 7 | Noise-aware measure + flat-region averaging | **Parity+** | averaging yes; noise-normalised measure, none |
| 8 | 16-bit pipeline — **done in 1.6.0** | **Catch-up** | 4 of 5 tools |
| 9 | Non-rigid / optical-flow refinement | **Differentiating** | none; Zerene explicitly global-only |
| 10 | Depth-driven blending — **done in 1.8.0** (depth-map *export* still open) | **Catch-up** | 4 of 5 tools |
| 11 | Focus-based ordering + bad-frame gating | **Catch-up** | Zerene auto-order |
| 12 | Retouching brush | **Catch-up** | both commercial tools |
| **13** | **Contrast / trust threshold** | **Catch-up** | Zerene, focus-stack, PICOLAY |
| **14** | **Depth-wise slabbing** | **Catch-up** | 3 of 5 tools |
| **15** | **Selectable focus operator** | **Parity+** | Shine Stacker only |
| **16** | **Contrast-weighted average method — done in 1.8.0** | **Catch-up** | Helicon A, Shine Stacker average |

Items 13-16 were identified by this comparison and were not in the original gap
analysis.

---

## 8. Where OpenFocus already leads

Stated plainly, because the tables above are weighted toward gaps.

- **Method breadth.** Nine fusion methods against two or three for the commercial
  tools.
- **Only tool with neural fusion.** StackMFF V4 and IFCNN refinement have no
  counterpart in any competitor here.
- **Only tool with block-DCT fusion.**
- **GPU acceleration across five methods** (CUDA/MPS), ahead of everything but
  focus-stack's OpenCL. Zerene has none.
- **Only tool accepting video input** for frame extraction.
- **Three alignment algorithms exposed and selectable** (SIFT-homography, ECC, and
  the 1.9.0 similarity-based scale / focus-breathing correction); competitors
  expose alignment as a set of toggles, not as choosable algorithms.
- **Measured, published quality audit.**
  [ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md) and the
  `tests/fusion_metrics.py` harness are something no competitor in this
  comparison publishes.

---

## 9. Reading of the comparison

The distinctive shape of the result: OpenFocus has **more fusion algorithms than
anyone and fewer supporting stages than anyone.**

The competitors converge on two blending methods each, then spend their remaining
effort on the pipeline around them — photometric normalisation, scale
correction, trust thresholds, slabbing, depth-map export, retouching. OpenFocus
inverts that; of the two families the field treats as mandatory it added the
Laplacian pyramid in 1.7.0 and the depth map — both selection rules — in 1.8.0, so
every mandatory blending family is now present.

With both mandatory families in place, the highest-value remaining work is no
longer another fusion method but the pipeline around them:

1. **Fix the inputs** — scale correction (§2) shipped in 1.9.0; photometric
   alignment (§3) is the remaining near-universal input fix. Both improve all nine
   existing methods at once.
2. **Finish the depth map** (§6) — the `Depth Map` method (1.8.0) turned the
   internal map into depth-driven blending, and 1.14.0 added halo suppression;
   still open are *exporting* it as a file and the trust thresholds and 3D
   output it feeds.

Then the differentiators — global label optimisation and non-rigid alignment —
where there is no field precedent to catch up to.

---

## Sources

Vendor and project documentation consulted for the closed-source and external
open-source entries:

- Helicon Focus — [Understanding the Focus Stacking Parameters](https://www.heliconsoft.com/helicon-focus-main-parameters/)
- Zerene Stacker — [DMap tutorial](https://zerenesystems.com/cms/stacker/docs/tutorials/tutorial003), [FAQ](https://zerenesystems.com/cms/stacker/docs/faqlist)
- Zerene Stacker — [PMax vs DMap community guide](https://macrobyraghu.com/2024/11/03/zerene-stacker-a-guide-to-pmax-and-dmap/), [alignment parameters discussion](https://photomacrography.net/forum/viewtopic.php?t=21508), [slabbing](http://extreme-macro.co.uk/zerene-slabbing/)
- focus-stack — [PetteriAimonen/focus-stack](https://github.com/PetteriAimonen/focus-stack)
- Shine Stacker — [documentation](https://shinestacker.readthedocs.io/en/latest/focus_stacking.html), [lucalista/shinestacker](https://github.com/lucalista/shinestacker)
- PICOLAY — [Understanding Stacking Parameters](https://www.picolay.de/workshop/Understanding_Stacking-Parameters.pdf)
- Forster, Van De Ville, Berent, Sage, Unser, "Complex Wavelets for Extended
  Depth-of-Field: A New Method for the Fusion of Multichannel Microscopy Images",
  *Microscopy Research and Technique*, 2004 — the basis of focus-stack.
