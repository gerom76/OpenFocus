# <img src="assets/OpenFocus.png" alt="OpenFocus Logo" width="120"> OpenFocus

OpenFocus delivers focus stacking quality that rivals commercial-grade software, while staying fully open source and easy to extend.

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/your-org/OpenFocus"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository" /></a>
</p>

## 📢 News

> [!NOTE]
> 🎉 **2026.07.30**: **JPEG XL, at the speed of the machine.** `imagecodecs` gives libjxl **one thread** unless it is told otherwise, and that is what every JPEG XL call here had been getting. It is now told otherwise. On a 12 MP 16-bit frame and 32 cores, saving a result went from **35 s to 2.6 s** and a single frame decodes in **0.10 s instead of 1.7 s** — the encoder takes the whole machine, and the loader divides the cores between its own decode workers and libjxl's pool, rounding up so a stack that does not divide evenly into the cores does not leave any idle. A stack load gains most where it has fewer frames than the machine has cores (4 frames: **3.4 s to 1.2 s**) and least where the frame count already fills it. A `.jxl` stack is also sized before it is loaded now, from a codestream-header reader written for the purpose (`imagecodecs` exposes none), so it gets the same up-front memory estimate and memory-bounded thread count as RAW — a libjxl decode peaks at about three times the frame it returns. There is no GPU path and there cannot be one: neither nvJPEG nor nvImageCodec implements JPEG XL.

> 🎉 **2026.07.30**: **DNG compression and fast-load previews.** *Settings → DNG Output* now controls how `.dng` results are written. **Compression** offers uncompressed (still the default), **lossless** — lossless Huffman JPEG, the only lossless codec DNG permits for 16-bit linear data, which stores the pixels identically at roughly half the size at 16-bit and a third at 8-bit — and **lossy** baseline JPEG with a quality setting, which DNG restricts to 8-bit and which is meant for proxies rather than masters. **Embed fast-load preview** stores a half-resolution JPEG rendering in a SubIFD, so browsers, Lightroom and raw converters draw a thumbnail from a few hundred kilobytes instead of decoding the full-resolution linear raw first. That is the DNG specification's own preview mechanism, not Adobe's proprietary "Fast Load Data" cache, which only Adobe's converter can produce. The lossless encoder is written in-house (`utils/ljpeg.py`), because no available library writes multi-component lossless JPEG and the single-component encoding the spec would allow is one LibRaw mis-decodes. The lossless *mode* is offered only where `imagecodecs` is installed to read it back — a result that cannot be reopened is not a result. Every mode is verified against LibRaw, and settings persist in `openfocus.cfg.json`.

> 🎉 **2026.07.30**: **DNG in and out.** `.dng` stacks load like any other source — camera and Adobe DNG Converter files are developed by LibRaw, alongside the NEF/NRW that already were. Results can be saved as DNG too, from the save dialogs, drag-out and the batch format list: an uncompressed **linear** (demosaiced) DNG at full 16-bit depth, tagged with sRGB primaries, an already-neutral white balance and the BT.709 transfer curve the develop actually applies, so Lightroom and RawTherapee render the result you actually saw rather than putting a second tone curve on top of it. Reloading one into OpenFocus gives back the exact pixels, because a DNG OpenFocus wrote is read straight from its strips instead of being developed twice. Writing needs nothing beyond numpy — OpenCV cannot write DNG and LibRaw cannot write at all, so the container is assembled directly.

> 🎉 **2026.07.29**: **JPEG XL in as well as out.** `.jxl` stacks now load like any other source — in the folder and file dialogs, on drag and drop, and in the folder-input paths of the fusion methods. libjxl does the decoding, so a 16-bit JPEG XL enters the pipeline at 16 bits instead of arriving pre-narrowed, and a `.jxl` source hands its EXIF on to the render the way a JPEG does, read straight out of the container's Exif box. Needs `pip install imagecodecs`; without it `.jxl` is simply not a supported input, exactly as RAW is not without rawpy.

> 🎉 **2026.07.29**: **WebP in and out.** `.webp` stacks load like any other source, and results can be saved as WebP from the save dialogs and the batch format list. Written losslessly, like every other format here, so a fused result is not quietly recompressed. It is an 8-bit container, so a 16-bit render is narrowed on the way out, and its 16383 px per side limit is reported as itself rather than as a failed save.

> 🎉 **2026.07.27**: **Pyramid, reworked and opened up.** The textbook choose-max rule copies the sharpest coefficient of every band and discards the rest, which has no answer for the parts of a frame where nothing is in focus — there it picks by grain, stitches the background out of frames that disagree, and reconstructs thin dark filaments no frame ever had. It now weights frames by how far they fall behind the best rather than picking one, compares them in units of their own grain, lets the frames carrying the detail carry the smooth base with it, and holds every pixel inside the range its own frames span. Every scenario in the test suite improves — **+4.5 dB** on a deep stack with a never-sharp background, **+16.7 dB** where a bright defocused veil used to win, and no invented pixels anywhere — and all six controls behind it are exposed in the right panel, saved, inherited by batch jobs and written into output filenames. See item 19 in [ALGORITHM_IMPROVEMENTS.md](docs/ALGORITHM_IMPROVEMENTS.md).

> 🎉 **2026.07.26**: **JPEG XL output.** Results can be saved as `.jxl` — lossless, typically half the size of the equivalent PNG, and 16-bit throughout, with the same EXIF/XMP metadata written into the container's boxes. Available in the save dialogs, drag-out and the batch format list; needs `pip install imagecodecs`, and is simply not offered without it.

> 🎉 **2026.07.26**: **Metadata on saved results.** A fused JPEG or PNG now inherits the **EXIF** block of the first source frame of the render — camera, lens and exposure survive the stack — and carries an **XMP** packet with two groups: `OpenFocus`, holding the program version, render date, render duration and **every option the render ran with** (registration stages and reference frame, fusion method, kernel and halo size, IFCNN, contrast, ROI, tiling, bit depth, processing unit), and `Camera`, every camera tag of the source cloned as plain text so the shot's settings read without an EXIF parser. Both are spliced into the encoded file, so pixels are never recompressed and 16-bit PNG stays 16-bit. Applies to single saves, drag-out and batch output.

> 🎉 **2026.07.24**: Selectable **Reference Frame** for registration — choose which frame the alignment chain is anchored to: *First* (frame 0, the historical default), *Middle*, or *Last*. Anchoring on the middle frame halves the longest chain, spreading accumulated drift symmetrically instead of piling it up at the far end of long stacks. The reference frame is held fixed (cropped, never warped); set it under **Settings → Registration**.

> 🎉 **2026.07.24**: New **Scale (focus breathing)** registration method — corrects the magnification change a lens introduces as the focus plane moves through a stack. Fits a constrained similarity (uniform scale + rotation + translation) via `estimateAffinePartial2D`/RANSAC, so it cancels size drift without the overfitting a full homography risks on blurred frames. Independent toggle that composes ahead of Homography/ECC; off by default.

> 🎉 **2026.07.23**: New **Depth Map** fusion method with two modes — *Max* (hard per-pixel select, the classic depth map) and *Average* (contrast-weighted blend that recovers multi-frame SNR in flat regions). Both are fully CPU-based and driven by a single focus-measure window. This closes the last two blending families the field treats as mandatory.

> 🎉 **2026.07.23**: New **Pyramid** fusion method — Laplacian-pyramid choose-max blending. Fully CPU-based with no tuning, it scores highest of all methods on our synthetic benchmark and makes a strong, dependable default for typical focus stacks. *(Reworked on 2026.07.27 — see above.)*

> 🎉 **2026.07.23**: Optional **contrast enhancement** for fused output. Pick *Auto* (color-safe global tone curve) or *Local* (CLAHE) with a strength slider — it applies after fusion and updates the preview live, so the stored result stays untouched. Off by default.

> 🎉 **2026.07.22**: End-to-end **16-bit pipeline**. Stacks whose sources carry more than 8 bits — RAW, 16-bit TIFF/PNG — are now loaded, aligned, fused and saved at full depth. Switch between Auto / 8-bit / 16-bit under *Settings → Bit Depth*.

> 🎉 **2026.07.21**: GFG-FGF fusion now runs on GPU (CUDA/MPS via PyTorch) with automatic CPU fallback — ~7x faster on large stacks.

> 🎉 **2026.07.20**: DTCWT fusion now runs on GPU (CUDA/MPS via pytorch_wavelets) with automatic CPU fallback — over 100x faster on large stacks.

> 🎉 **2026.07.19**: Guided Filter and DCT fusion now run on GPU (CUDA/MPS via PyTorch) with automatic CPU fallback — up to ~5x faster on large stacks.

> 🎉 **2026.07.19**: Image stacks now load in parallel across CPU cores (~4x faster for RAW and large stacks).

> 🎉 **2026.07.19**: Added support for loading Nikon RAW images (NEF/NRW) via rawpy (requires `pip install rawpy`).

> 🎉 **2026.01.13**: Optimized ROI mode processing and fixed bugs to improve performance and stability.
 
> 🎉 **2026.01.12**: Added drag-and-drop image import on Mac and refactored core modules for improved code maintainability and readability.

> 🎉 **2026.01.10**: Added bilingual support, a status dashboard, and new ROI fusion options to improve efficiency and flexibility.

> 🎉 **2026.01.09**: Improved UI and navigation, faster parallel processing, multi-folder batch support, and bug fixes.

> 🎉 **2025.12.11**: Added functionality to read image stacks in video format.
 
> 🎉 **2025.12.11**: Thanks to Rangj for providing the C++ implementation of the GFG-FGF fusion algorithm, which is now available in the software.

> 🎉 **2025.12.11**: We have fixed some bugs and added configuration options such as block-wise fusion to avoid OOM (Out of Memory) issues.

> 🎉 **2025.12.05**: OpenFocus officially released — welcome to try it.

<a id="environment-setup"></a>
## ⚙️ Environment Setup
```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install opencv-python pyqt6 numpy imageio dtcwt scipy torch torchvision 
python main.py
```

> **Pre-built package (Windows only):** Grab the compact Windows build from the [Releases](https://github.com/Xinzhe99/OpenFocus/releases) page; other platforms can run from source.
## Table of Contents
- [⚙️ Environment Setup](#environment-setup)
- [🔭 Overview](#overview)
- [✨ Highlights](#highlights)
- [🧪 Fusion & Registration Methods](#fusion--registration-methods)
- [📚 References](#references)
- [🤝 Contribution](#contribution)
- [📄 License](#license)

<a id="overview"></a>
## 🔭 Overview
OpenFocus is a PyQt6-based multi-focus registration and fusion workstation that delivers commercial-grade alignment and blending results. The project is fully open source (MIT License) and runs on CPU by default with optional GPU acceleration for the StackMFF V4 neural model.

<p align="center">
	<img src="assets/ui.jpg" alt="OpenFocus UI" width="720">
</p>

<a id="highlights"></a>
## ✨ Highlights
- **Beginner-Friendly**: Plug-and-play workflows with unapologetically simple, guided operations.
- **Flexible Processing Flows**: Run fusion-only, registration-only, or combined registration + fusion pipelines depending on your workload.
- **Batch Automation**: Kick off batch jobs across multiple folders with live progress, cancellation, and automatic output organization.
- **Annotation & Export Toolkit**: Overlay labels, export GIF animations, and save processed stacks in JPG/PNG/BMP/TIFF/WebP/JXL/DNG with consistent metadata handling.
- **Metadata Passthrough**: A saved JPG/PNG/JXL result keeps the EXIF of the first source frame, clones its camera tags into a readable XMP `Camera` section, and records the OpenFocus version, render date, render duration and the full set of options the render used in an `OpenFocus` group — enough to reproduce the render, all added without recompressing the image.
- **AI-Assisted Fusion**: Ship with StackMFF V4 to unlock deep-learning-quality fusion alongside classic signal-processing methods.
- **16-bit Processing**: RAW and 16-bit TIFF/PNG/JXL stacks stay at full depth from load through alignment, fusion and export — no banding in smooth gradients, and headroom left for blending. Auto-engages on >8-bit sources, or force 8/16-bit from *Settings → Bit Depth*.
- **Contrast Enhancement**: Optional post-fusion tone control — *Auto* (color-safe global curve) or *Local* (CLAHE) with a strength slider. Applies after fusion and previews live without re-rendering, so it never alters the stored result. Color-safe and depth-aware.

<a id="fusion--registration-methods"></a>
## 🧪 Algorithms
### Fusion Algorithms

- **Guided Filter**: Fast edge-preserving fusion that enhances contrast while suppressing noise.
- **DCT Multi-Focus Fusion**: Frequency-domain technique optimized for crisp detail recovery.
- **Dual-Tree Complex Wavelet Transform (DTCWT)**: Multi-scale representation that preserves fine texture structures.
- **GFG-FGF**: GFG-FGF is based on a generalized four-neighborhood Gaussian gradient (GFG) operator combined with a fast guided filter (FGF). 
- **Pyramid**: Laplacian-pyramid fusion. Band by band, each frame is weighted by how far its local energy falls behind the best on offer, so a sharp frame wins outright while frames that tie — a background no frame resolves — are averaged rather than picked between, and the result is held inside the range its own frames span. Six tuning controls, all defaulted to their measured best. A strong general-purpose default.
- **Depth Map**: Per-pixel depth-map fusion from a local Laplacian focus measure, in two modes. *Max* takes each pixel whole from the sharpest frame (an order-independent hard select that keeps colour and noise clean within a slice); *Average* blends frames by their focus measure, so flat regions collapse to the mean and recover the stack's multi-frame SNR (√N noise reduction) while detail still follows the sharpest frame.
- **StackMFF V4**: Pretrained deep model delivering state-of-the-art focus stacking quality.

### Registration Algorithms

- **Scale (focus breathing)**: Corrects the magnification change a lens introduces as the focus plane moves through a stack. Fits a constrained similarity (uniform scale + rotation + translation) via `estimateAffinePartial2D`/RANSAC on keypoint matches, so it cancels size drift without the overfitting a full homography risks on frames that differ in blur.
- **Homography**: Performs feature-based projective alignment using keypoint matching and RANSAC to handle global perspective transformations.
- **ECC**: Performs intensity-based alignment by maximizing the enhanced correlation coefficient for precise, sub-pixel registration.

Registration stages are independent and compose in pipeline order: **Scale → Homography → ECC**. Every stage is anchored to a selectable **reference frame** — *First* (default), *Middle*, or *Last* — letting you spread accumulated chain drift instead of concentrating it at one end of the stack.
  
> **License Notice:** Every fusion/registration algorithm included comes from open-source research implementations. When using or redistributing them, please follow each algorithm’s original license terms in addition to the OpenFocus MIT license.

<a id="references"></a>
## 📚 References
- M. B. A. Haghighat, A. Aghagolzadeh, and H. Seyedarabi, "Multi-focus image fusion for visual sensor networks in DCT domain," *Computers & Electrical Engineering*, vol. 37, no. 5, pp. 789-797, 2011.
- J. J. Lewis, R. J. O'Callaghan, S. G. Nikolov, D. R. Bull, and N. Canagarajah, "Pixel- and region-based image fusion with complex wavelets," *Information Fusion*, vol. 8, no. 2, pp. 119-130, 2007.
- S. Li, X. Kang, and J. Hu, "Image fusion with guided filtering," *IEEE Transactions on Image Processing*, vol. 22, no. 7, pp. 2864-2875, 2013.
- P. J. Burt and E. H. Adelson, "The Laplacian pyramid as a compact image code," *IEEE Transactions on Communications*, vol. 31, no. 4, pp. 532-540, 1983.
- 付宏语, 巩岩, 汪路涵, 等. 多聚焦显微图像融合算法[J]. Laser & Optoelectronics Progress, 2024, 61(6): 0618022-0618022-9.

<a id="contribution"></a>
## 🤝 Contribution
We welcome community contributions of all kinds:
1. **Issues**: Report bugs, request features, or propose UX enhancements.
2. **Algorithm & Performance Work**: Share new fusion/registration ideas, optimizations.

> Bug reports or suggestions? Please open an issue so we can follow up quickly.

<a id="license"></a>

## 📄 License
This project is released under the [MIT License](./LICENSE). Feel free to use, modify, and distribute within the terms of the license.

If you publish images created with OpenFocus, please consider adding a note such as:

Created with OpenFocus – https://github.com/Xinzhe99/OpenFocus

This is not mandatory, but highly appreciated.

<p align="center" style="font-size:1.25rem; font-weight:600;">
  If OpenFocus helps you, please consider leaving a ⭐ on the repository!
</p>














