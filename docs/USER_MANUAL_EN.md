# OpenFocus User Manual

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Main Interface Overview](#3-main-interface-overview)
4. [Importing Images and Video](#4-importing-images-and-video)
5. [Viewing and Navigating Images](#5-viewing-and-navigating-images)
6. [Image Registration](#6-image-registration)
7. [Image Fusion Methods](#7-image-fusion-methods)
8. [Image Transformations](#8-image-transformations)
9. [Labels and Annotations](#9-labels-and-annotations)
10. [Settings and Configuration](#10-settings-and-configuration)
11. [Batch Processing](#11-batch-processing)
12. [Exporting Results](#12-exporting-results)
13. [Keyboard Shortcuts](#13-keyboard-shortcuts)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Introduction

OpenFocus is a professional multi-focus image fusion desktop application designed for researchers, engineers, and image processing professionals. The software processes multi-focus image sequences using advanced registration and fusion algorithms to generate all-in-focus images with full depth of field.

### Key Features

- **Multi-Focus Image Fusion**: Combine multiple images with different focus points into one fully focused image
- **Multiple Fusion Algorithms**: Choose from Guided Filter, DCT, DTCWT, GFG-FGF, Pyramid, Depth Map (Max / Average), and StackMFF-V4 (deep learning)
- **Image Registration**: Align misaligned image sequences using Scale (focus breathing), ECC, or Homography methods
- **Batch Processing**: Process multiple image folders simultaneously
- **Flexible Export**: Save results as individual images, folders, or GIF animations
- **Image Transformations**: Rotate, flip, and resize image stacks
- **Label Addition**: Add text labels to registered and input stacks

### Supported Input Formats

- Image files: JPG, PNG, BMP, TIFF, WebP, JXL (JPEG XL, with `imagecodecs` installed), and other common formats
- RAW files: NEF, NRW, DNG (developed by LibRaw, with `rawpy` installed)
- Video files: MP4 format (automatically decoded into image sequences)
- Folders containing numbered image sequences

### Supported Output Formats

- Individual images: JPG, PNG, BMP, TIFF, JXL (JPEG XL, with `imagecodecs` installed), DNG
- Image sequences: Saved as folders
- Animations: GIF format

---

## 2. Getting Started

### Starting the Application

After installation, launch OpenFocus by:

1. **Windows (Executable)**: Double-click the OpenFocus shortcut or executable file
2. **Python Source**: Run `python main.py` in your activated conda environment

Upon launch, you will see the main application window with a dark theme interface.

### Quick Start Workflow

1. Import your image sequence (File → Open Folder or drag-and-drop)
2. Configure fusion settings in the right panel
3. Optionally enable registration for alignment
4. Adjust kernel size if needed
5. Click "Start Render" to begin processing (or "Stop" to interrupt a running render)
6. Export your results

---

## 3. Main Interface Overview

The OpenFocus interface consists of three main areas:

### Left Panel: Image Display Area

The left portion of the window displays your images in a split view:

- **Source Image Panel (Top)**: Shows the input image sequence
- **Result Image Panel (Bottom)**: Shows the fusion or registration results
- **Navigation Slider**: Use the slider below each panel to scroll through image frames
- **Image Info Labels**: Display current frame position and image dimensions

You can resize the panels by dragging the splitter handle between them.

### Right Panel: Control Panel

The right panel contains all configuration options organized vertically:

1. **Fusion Settings**: Select fusion algorithm and kernel size
2. **Registration Settings**: Enable/disable alignment methods
3. **Action Buttons**: Reset defaults, Start Render, and Stop (interrupts an in-progress render)
4. **Source Images List**: Shows all loaded images with filenames
5. **Output List**: Shows generated results

### Menu Bar

The top menu bar provides access to all functions:

- **File**: Import/export operations
- **Edit**: Image transformations and labels
- **Batch**: Batch processing
- **Settings**: Tile, Registration, and Thread configurations
- **Help**: Environment info and contact information

---

## 4. Importing Images and Video

### Opening an Image Folder

1. Go to **File → Open Folder** or press `Ctrl+O`
2. Select a folder containing your image sequence
3. A downsample dialog will appear - choose either a scale factor (%) or a target size for the longer edge (px)
4. Click OK to load the images

The longer-edge target is applied to each image separately, so a mixed-size
stack ends up with one common long edge. Images already at or below the target
are left untouched - downsampling never enlarges an image.

The software will automatically:
- Detect and sort image files by filename
- Display the first image in the source panel
- Populate the source images list with all frames

### Opening Video Files

1. Go to **File → Open Video** or press `Ctrl+Shift+O`
2. Select an MP4 video file
3. The video will be automatically decoded into individual frames
4. Each frame becomes part of the image stack

### Drag and Drop

You can also drag a folder directly onto the application window to import images.

### Clearing the Stack

To remove all loaded images, go to **File → Clear Stack** or press `Ctrl+W`.

---

## 5. Viewing and Navigating Images

### Navigating the Image Stack

- **Slider**: Drag the slider below the source panel to scroll through frames
- **File List**: Click any filename in the right panel's source images list
- **Keyboard**: Use arrow keys (Left/Right) to navigate

### Zooming and Panning

- **Zoom In/Out**: Mouse wheel over the image
- **Fine Zoom**: Hold `Ctrl` + mouse wheel for smaller increments
- **Fit to Window**: Double-click the image to toggle between fit and 100% view
- **Pan**: Click and drag to move the image within the panel

### Synchronized Navigation

When you navigate in the source panel, the result panel shows the corresponding result frame if available.

---

## 6. Image Registration

Image registration corrects spatial misalignment between frames in your image stack. This is essential when images have slight shifts or perspective changes.

### Registration Methods

#### Scale (focus breathing)

- Corrects the magnification change a lens introduces as the focus plane moves
  through the stack — the effect known as *focus breathing*
- Fits a similarity transform (uniform scale + small rotation + recentring) from
  feature matches, so it cancels size drift without the overfitting a full
  homography risks on frames that differ in blur
- Runs first, as a coarse magnification correction, and composes with ECC and
  Homography
- Use it when frames grow or shrink slightly from the first shot to the last

#### ECC (Enhanced Correlation Coefficient)

- Uses optimization to find the best alignment based on correlation
- Suitable for subtle misalignments
- Provides sub-pixel accuracy
- Slower than Homography but more precise

#### Homography

- Based on feature point matching (SIFT/ORB features)
- Handles larger geometric transformations
- Faster than ECC for initial alignment
- Good for images with significant perspective changes
- Fits a perspective transform and a constrained similarity to each frame pair
  and keeps the perspective one only where it predicts matches it was not
  fitted to, so a stack shot on a rail - which has no perspective in it - is
  not bent to fit feature-matching noise

### Enabling Registration

1. In the right panel, locate the **Registration** group
2. Check **Scale (focus breathing)** to correct magnification drift
3. Check **Align (ECC)** to enable ECC alignment
4. Check **Align (Homography)** to enable Homography alignment
5. Methods are independent; when several are on they run in order (Scale →
   Homography → ECC)

### When to Use Registration

- Your images have visible misalignment
- Images were captured handheld
- The scene has parallax (closer objects shift between frames)
- Frames change in magnification between the first and last shot (focus breathing)
- Results show ghosting or doubling artifacts

### Registration Settings

Access additional settings via **Settings → Registration**:
- **Downscale Width**: Control preprocessing resolution (default: 1024px)
- Lower values = faster processing, potentially lower accuracy
- Higher values = slower processing, potentially higher accuracy
- Setting it to the full width of your frames aligns at full resolution. Before
  1.30.6 the setting was ignored on frames of 2048px or more.

#### Reference Frame

Registration aligns each frame to its neighbour and chains those transforms back
to a single fixed frame — the *reference*. Alignment error accumulates with
distance from the reference, so on long stacks the frames farthest from it drift
the most. Choose which frame stays fixed:

- **First** (default): anchors on frame 0, matching earlier versions. Error
  grows toward the end of the stack.
- **Middle**: anchors on the centre frame, halving the longest chain so residual
  drift is spread symmetrically across the stack. Best for long stacks where the
  far end drifts or shows ghosting.
- **Last**: anchors on the final frame — the mirror of **First**, useful when the
  sharpest or most important detail sits at the end of the stack.

The reference frame is held fixed (it is only cropped to the shared valid
region, never warped); every other frame is aligned onto it.

---

## 7. Image Fusion Methods

OpenFocus offers eight fusion algorithms. Each has different characteristics suitable for various image types.

### Guided Filter (Default)

- **Algorithm**: Uses guided filtering for edge-preserving fusion
- **Best for**: General purpose, balanced speed and quality
- **Advantages**: Good edge preservation, fast processing
- **Parameter**: Kernel size (adjustable from 1-51, odd values only)
  - Smaller kernel: More local detail, potentially more noise
  - Larger kernel: Smoother results, better noise suppression

### DCT (Discrete Cosine Transform)

- **Algorithm**: Frequency-domain fusion using DCT coefficients
- **Best for**: Texture-rich images, scientific imaging
- **Advantages**: Preserves fine textures, mathematically sound
- **Parameter**: Kernel size controls the processing window

### DTCWT (Dual-Tree Complex Wavelet Transform)

- **Algorithm**: Multi-scale wavelet domain fusion
- **Best for**: Complex scenes with multiple detail levels
- **Advantages**: Excellent multi-scale analysis, good directionality
- **Parameter**: Kernel size for filtering operations

### GFG-FGF (Generalized Four邻域 Gradient - Fast Guided Filter)

- **Algorithm**: Gradient-based fusion with fast guided filtering
- **Best for**: Images with clear focus regions
- **Advantages**: Fast, good focus region detection
- **Parameter**: Kernel size adjustment

### Pyramid (Laplacian Pyramid)

- **Algorithm**: Laplacian-pyramid fusion — band by band, each frame is weighted by how far its local focus energy falls behind the best on offer, so a sharp frame wins outright and frames that tie are averaged rather than picked between
- **Best for**: A strong, dependable general-purpose default across most focus stacks, and the best of the classical methods on deep stacks with a background that is never sharp
- **Advantages**: Sharp, seam-free results; fully CPU-based (GPU optional); every pixel is held inside the range its own frames span, so the render cannot invent structure
- **Parameters**: Kernel slider (energy window), plus Pyramid levels, Selectivity, Scale coherence, Base band, Grain estimate, and two switches — see *Pyramid tuning* below. The defaults are the measured best; nothing has to be touched

### Depth Map (Max / Average)

- **Algorithm**: Per-pixel depth-map fusion from a local Laplacian focus measure, in two modes
  - **Max**: takes each pixel whole from the frame with the highest focus measure — the classic hard depth map, an order-independent per-pixel select that keeps colour and noise clean within a slice
  - **Average**: blends frames by a power of their focus measure, so flat regions collapse to the plain mean and recover the stack's multi-frame SNR (a free √N noise reduction), while sharp detail still follows the frame that holds it. The power is set by the *Selectivity* dial below and is what makes that second half true however deep the stack is
- **Best for**: *Max* — clean, artifact-free selection on well-defined subjects; *Average* — stacks with large smooth areas where a hard select would chase sensor noise
- **Advantages**: Runs on GPU (CUDA/MPS) with automatic CPU fallback; no colour splitting across sources; order-independent
- **Background**: See [how_depthmap_quality_works.html](how_depthmap_quality_works.html) for the three defects fixed between 1.32.0 and 1.34.0 — what each one looked like, the mechanism behind it, which mode it reached, and the measurements the fixes were chosen on
- **How it works**: See [algorithms/how-depthmap-works.html](algorithms/how-depthmap-works.html) for the method end to end — the focus measure step by step, both selection rules, what the implementation guarantees about determinism, memory and device parity, and a dial-by-dial comparison with Helicon Focus Methods A and B (including how their *Radius* converts to this *Kernel*)
- **Note**: In *Max* mode the depth map is despeckled before any pixels are gathered. Over an area no frame ever resolves the focus measure has no real winner, and an undespeckled selection tears such an area into a mosaic of patches taken from frames that look nothing alike; the despeckle makes those pixels follow their neighbourhood instead. Genuinely sharp detail already agrees with its neighbours, so it is not what pays for this
- **Parameter**: Kernel size sets the window the focus measure is pooled over (larger is steadier on noise, smaller follows finer detail). The measure is pooled at two scales at once — the window you set, and a narrow one — so a sharply focused outline can no longer claim the background beside it on the strength of energy that lives half a window away. That is what used to draw a flat, washed-out ring around every subject at large kernel sizes, and large kernels are now safe to use
- **Parameter**: Halo suppression radius (0 = off). A defocused foreground edge casts a bright glow over the background in the frames where the background is sharp, and plain per-pixel selection copies that glow into the result — the classic focus-stacking halo. With a radius set, a sharply focused region also claims the surrounding band its glow contaminates, so the ring comes out as natural defocused background instead. **Leave this at 0 unless a glow actually survives**: the two-scale measure already discounts a veil on its own (a glow is low-frequency, so it scores badly on the narrow window), and the dilation does not remove a ring so much as fill one with defocused pixels, whether or not there was a glow to fight. If you do need it, set it to roughly the visible halo width — larger radii round off genuine detail near depth edges, the same trade-off the Radius dial has in Helicon Focus and Zerene Stacker. **The slider's ceiling follows the kernel size**, because a radius the pooling window cannot speak for stops suppressing haloes and starts filling a band around *every* contour in the frame with defocused pixels — soft blobs roughly twice the radius across, hugging every edge, dust speck and silhouette. Measured on a 333-frame stack at kernel 5, share of the frame moved against the same render with the dial off: 8.5% at radius 2, 18.3% at 5, 24.8% at 9, 29.9% at 15. Nothing about the scene changed; only the radius did. A radius dialled in at a wide kernel therefore comes down when you narrow the kernel, rather than staying as the same absolute band on a third of the evidence
- **Parameter** (*Max* only): Depth coherence (0-100, default 50). How readily a pixel's own focus decision is given up as unfounded, with the depth of every pixel given up on interpolated from the ones that were not. Over a region no frame ever resolves — a background behind the sweep, a dark flat patch — every frame measures the same and the winner is decided by grain, so neighbouring pixels take frames from opposite ends of the stack and the region tears into a mosaic of hard-edged patches. At 0 that is exactly what you get, byte for byte with the pre-1.35.0 hard select. Raising it declines to select over more of the frame and fills those areas from what their neighbourhood measured instead. Measured on a 333-frame stack against another program's render of the same capture, the fine-detail gap closes from 0.51 at 0 to 0.24 at 50, and the grain left in detail-free areas from 4.4 to 3.1 — but agreement on *where* the detail is peaks at 50 and falls away above it, because past there the fill starts flattening depth structure that is real and the subject loses its own surface texture. Raise it if a smooth background is blotchy; lower it if a surface has gone waxy
- **Parameter** (*Average* only): Selectivity (0-100, default 50). How sharply the blend favours the frame that actually holds the detail. Left linear (0), the blend only selects while the stack is short: a defocused frame still measures a fraction of the peak, and a deep stack adds that fraction up once per frame until it buries the one frame that resolved the pixel. On a 333-frame stack the sharpest frame was contributing under 2% of the output at the median pixel and the result was effectively the arithmetic mean of every frame - which is one enormous defocus kernel, and reads as a veiled, low-contrast image with lifted blacks that got worse the deeper the stack. Raising the dial makes the weight super-linear, so the defocused frames can no longer outvote the sharp one by sheer count. It costs nothing where averaging is genuinely the right answer - where every frame measures the same, every weight still comes out equal - so what you trade is some of the multi-frame noise reduction in those regions. Raise it if the result still looks veiled; lower it if a smooth background has gone grainy
- **Parameter** (*Max* only): Edge coherence radius (0 = off, ceiling 12). The complement to Depth coherence above, and it reaches what that one by design cannot. The fill is gated on how much the focus measure had to say, so it acts hardest where it said least and declines to act over the regions it found *merely adequate* — and those are exactly where the depth map is noisiest without being obviously wrong. Measured against another program's depth map of the same capture, tile-by-tile agreement in the busiest fifth of the frame is 0.93 and in the middle fifths only 0.35–0.42; no setting of any other dial moves them. This one filters the finished depth map against the all-in-focus picture, so the depth is averaged *within* a surface and left stepping at an occlusion — which a plain blur of the same reach cannot do without rounding the step off too. On a 333-frame stack, agreement rises from 0.962 unfiltered to 0.970 at radius 8, while the fine-detail gap falls from 0.24 to 0.15. **Keep it at or under the kernel size**: at radius 16 on a kernel-25 render it is reaching over real depth steps and scores below the unfiltered baseline, which is why the slider stops at 12. Raise it if a surface that should be smooth is speckled with pixels from the wrong slice
- **Parameter** (*Average* only): Weight coherence radius (0 = off). Passes each frame's share of the blend through an edge-aware filter before the pixels are gathered. At any selectivity high enough to keep a deep stack from hazing, the blend is a hard select in all but name, and over a region no frame resolves that choice is decided by grain: neighbouring pixels draw the same smooth surface from frames eight slices apart, which do not carry the same local brightness, and the region breaks into blotches. Filtering the shares averages that choice where the picture is featureless and leaves it alone where it is not. The reach is the whole risk — it assumes depth varies smoothly out to the radius set, which pays on a deep capture of a real surface and costs where the correct frame changes every few pixels. Measured best at 16 on a 333-frame macro capture and at 0-4 on synthetic stacks that step depth every 27 px. It costs a second pass over the stack
- **Parameter** (both modes): Slice coherence radius (0 = off). Pooling along the *frame* axis rather than across the picture, and the one dial the two modes share. A stack that oversamples its depth of field resolves each pixel about equally well in the several frames around its focus peak, which differ mostly in their grain, so averaging across that band divides the grain by roughly √N at no cost in sharpness — which no spatial setting can offer. In *Average* it pools each pixel's weight across neighbouring frames; in *Max* it renders each pixel from a band of frames around its own depth instead of from the single winner, which is the only way that mode can divide grain at all, since every other dial there moves the depth map while the pixel still comes from one frame whole. **Set it from the capture's sampling, not from the picture**: about half the width of the focus peak, so 4-5 on a sweep that oversamples its depth of field sevenfold and **0** on one that steps a full depth of field per frame, where the neighbouring frames are the ones that resolve the pixel *worst*. Overshooting is not subtle — at 8 slices on the capture it was measured on, every part of the frame got worse at once. **Leave it at 0 on an unregistered stack**: it is the only dial here that averages different frames into one pixel, so it needs them to agree on where that pixel is. On *Max* it is free (the render already walks the whole stack); on *Average* it shares the extra pass with Weight coherence, so having both on costs no more than either

### StackMFF-V4 (Deep Learning)

- **Algorithm**: Neural network-based fusion (requires PyTorch)
- **Best for**: Highest quality results when GPU is available
- **Advantages**: State-of-the-art fusion quality, automatic optimization
- **Requirements**: PyTorch installation, optional CUDA GPU
- **Note**: May be unavailable if PyTorch is not installed

### Pyramid Tuning

Selecting **Pyramid** reveals its own group of controls under the kernel slider.
Every one of them is a trade rather than a right answer, which is why they are
exposed; the defaults are what measured best across the project's test stacks,
so leaving them alone is a valid choice.

| Control | Default | What it does |
|---|---|---|
| Kernel slider | 5 | The window each band's focus energy is pooled over before frames are compared. Small follows fine detail and can chase grain; large decides region by region and rounds off narrow in-focus structures. Small by design — the pyramid pools again at every level, so this window covers twice as much picture at each one |
| Pyramid levels | Auto | How many band-pass levels each frame is split into. Auto is five, limited by the image size. Fewer decide focus on coarser structure; more separate scales finely and cost time |
| Selectivity | Balanced | How sharply each band favours the sharpest frame. **Balanced** and **Strict** let a genuinely sharp frame win outright while frames that tie are averaged. **Average** and **Soft** blend more of the stack in — smoother backgrounds, softer detail. **Winner takes all** is the textbook rule: it takes the single best coefficient everywhere, including in defocused areas where the best is decided by grain, which is what leaves thin dark streaks across smooth backgrounds. Raising this above Balanced sharpens grain faster than it sharpens the picture on a deep stack, so **Strict** is worth reaching for on a clean, well-lit subject and not on a noisy one |
| Scale coherence | Off | Makes the coarse levels guide the fine ones, so a pixel cannot take its fine detail from one frame and its coarse structure from another. Reach for it if detail looks like it is sitting on the wrong background. It costs sharpness where near meets far, which is why it is off |
| Base band | Balanced | How hard the smooth, low-frequency layer follows the frames that won the detail. **Mean** averages every frame and hazes the result when most of the stack is defocused; **Balanced** follows the sharp frames hardest |
| Ignore grain when nothing is sharp | On | Compares frames in units of their own noise, so a bright, grainy, completely defocused frame cannot win the areas where nothing is in focus and stamp its flat tone over them |
| Grain estimate | Balanced | How much of each band the switch above reads that noise level off — the share of the picture it assumes holds nothing at that scale. **Minimal** takes it from the quietest corner, which understates the grain over the rest of a frame that is not evenly lit; **Broad** takes it from pixels that are resolving something, and divides part of that away with the grain. Greyed out when the switch above is off, because then no estimate is taken at all. Reach for **Broad** on a stack that is mostly empty background, and **Low** on one where the subject fills the frame |
| Keep pixels within the source range | On | Holds every pixel between the darkest and brightest value the frames actually have there. Rebuilding an image from bands taken out of different frames can otherwise produce values no frame had — the thin dark filaments over a smooth background this prevents |

**If a smooth, defocused area comes out crossed by thin dark strokes**, both
switches at the bottom are the ones that remove them, and both are on by
default. Check they have not been turned off, leave Selectivity at Balanced or
lower rather than Winner takes all, and if the strokes survive that, try Scale
coherence at Light or Medium.

Batch jobs inherit whatever the main window is set to, and any control moved off
its default is written into the output filename, so two renders that differ only
in tuning cannot overwrite each other.

### Selecting a Fusion Method

1. In the right panel, locate the **Fusion** group
2. Select one or more methods by checking the corresponding boxes
3. Adjust kernel size if applicable
4. Click **Start Render** to process

> **Stopping a render**: While a render is running, the **Stop** button next to Start Render becomes active. Click it to interrupt processing — the render unwinds at its next safe checkpoint (between pipeline stages, or between tiles on large images), the controls unlock, and no result is produced. For large tiled images this may take a moment while the tiles already in progress finish.

### Algorithm Comparison Guide

| Algorithm | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| Guided Filter | Fast | Good | General use |
| DCT | Medium | Good | Textures |
| DTCWT | Medium | Very Good | Complex scenes |
| GFG-FGF | Fast | Good | Focus regions |
| Pyramid | Fast | Very Good | Dependable default |
| Depth Map (Max) | Fast | Very Good | Clean hard select |
| Depth Map (Average) | Fast | Good | Smooth/noisy stacks (SNR) |
| StackMFF-V4 | Slow (GPU) | Excellent | Best quality |

---

## 8. Image Transformations

### Rotation

1. Go to **Edit → Rotate**
2. Choose rotation type:
   - **90° Clockwise**: Rotate all images 90 degrees right
   - **90° Counter-Clockwise**: Rotate all images 90 degrees left
   - **180°**: Flip images upside down

### Flipping

1. Go to **Edit → Flip**
2. Choose flip type:
   - **Horizontal Flip**: Mirror left-to-right
   - **Vertical Flip**: Mirror top-to-bottom

### Resizing

1. Go to **Edit → Resize**
2. A dialog will appear for resizing options
3. Enter new dimensions or scale percentage
4. All images in the stack will be resized proportionally

### Transformation Order

Transformations are applied to the entire image stack, maintaining alignment between frames.

---

## 9. Labels and Annotations

### Adding Labels

1. Ensure you have rendered results or loaded images
2. Go to **Edit → Add Label**
3. Enter your label text in the dialog
4. Labels are added to both registered stack and input stack

### Deleting Labels

- **Delete Registered Stack Labels**: Removes labels from aligned images
- **Delete Input Stack Labels**: Removes labels from original images

Access these options via **Edit → Delete Label** submenu.

---

## 10. Settings and Configuration

### Tile Settings (Memory Optimization)

For large images, OpenFocus uses tile-based processing to avoid memory issues.

Access via **Settings → Tile**:

- **Tile Block Size**: Size of each processing tile (default: 1024px)
- **Tile Overlap**: Overlap between tiles for seamless blending (default: 256px)
- **Tile Threshold**: Image size above which tiles are used (default: 2048px)

**Recommendations**:
- Smaller block size: Lower memory usage, slower processing
- Larger block size: Higher memory usage, faster processing
- Increase overlap if you see visible seams in results

### Registration Settings

Access via **Settings → Registration**:

- **Downscale Width**: Preprocessing resolution for registration
  - Default: 1024px
  - Lower for speed, higher for accuracy — honoured at every image size since
    1.30.6; values above the frame width mean full-resolution alignment
- **Reference Frame**: Frame held fixed during alignment — **First** (default),
  **Middle**, or **Last**. Middle minimises accumulated drift on long stacks.

### Thread Count Settings

Control CPU parallel processing:

Access via **Settings → Thread Count Settings**:

- Set the number of worker threads (default: 4)
- Match your CPU core count for optimal performance
- Higher values = faster processing, more CPU usage

---

## 11. Batch Processing

Process multiple folders automatically:

1. Go to **Batch → Batch Processing** or click the batch option
2. In the batch dialog:
   - **Add Folders**: Select folders containing image sequences
   - **Remove Folders**: Remove selected folders from the list
   - **Clear All**: Remove all folders
3. Configure output settings:
   - **Save to Subfolder**: Creates results in each input folder
   - **Save to Same Folder**: Overwrites or saves alongside input
   - **Custom Folder**: Specify output location
4. Review current fusion/registration settings
5. Click **Start Batch** to begin

The batch dialog shows real-time progress. You can cancel processing at any time.

### Batch Processing Notes

- Uses the same fusion and registration settings as manual processing
- Each folder is processed independently
- Filenames include timestamp and parameters for identification

---

## 12. Exporting Results

### Saving a Single Result

1. Select the result you want to save in the output list
2. Go to **File → Save** or press `Ctrl+S`
3. Choose format and location
4. Click Save

### Saving the Registered Stack

1. Go to **File → Save Stack → Registered Stack**
2. Choose save format:
   - **Save as Folder**: Saves each aligned frame as separate image files
   - **Save as GIF**: Creates an animated GIF of the aligned sequence

### Saving the Input Stack

1. Go to **File → Save Stack → Input Stack**
2. Choose save format:
   - **Save as Folder**: Asks which image format to write, then saves the original frames as separate images under their own names. The chosen format is remembered and offered first the next time you save
   - **Save as GIF**: Creates animated GIF of input sequence

### Export Formats

- **JPG**: Compressed, good quality-to-size ratio
- **PNG**: Lossless, supports transparency
- **BMP**: Uncompressed, maximum quality
- **TIFF**: High quality, supports layers
- **JXL** (JPEG XL): Lossless and much smaller than PNG, keeps 16-bit depth and carries the EXIF/XMP metadata. Requires `pip install imagecodecs`; without it the format is not offered
- **DNG**: Linear (demosaiced) DNG, always 16-bit. Written for raw-converter workflows: the samples are put through the BT.709 curve on the way out so the file really is scene-linear, and it carries sRGB primaries, an already-neutral white balance and its own camera profile. Lightroom, RawTherapee or Luminar then renders it as the result you saw rather than a brighter, flatter one. Reloading it into OpenFocus returns the frame that was saved. The source camera's EXIF — lens, exposure, MakerNote — is carried across, but the `Make` and `Model` stay OpenFocus's own, because those are what a converter reads to decide which camera profile to offer. Compression, the fast-load preview and the colour space are set under **Settings → DNG Output**. See [how_dng_export_works.html](how_dng_export_works.html) for what each option changes, with pictures
- **GIF**: Animation format

---

## 13. Keyboard Shortcuts

### File Operations

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open Folder |
| `Ctrl+Shift+O` | Open Video |
| `Ctrl+S` | Save Result |
| `Ctrl+Shift+S` | Save Registered Stack |
| `Ctrl+W` | Clear Stack |
| `Ctrl+Q` | Exit |

### Navigation

| Shortcut | Action |
|----------|--------|
| `Left Arrow` | Previous frame |
| `Right Arrow` | Next frame |
| `Mouse Wheel` | Zoom in/out |
| `Ctrl + Wheel` | Fine zoom |
| `Double-click` | Toggle fit/100% view |

### General

| Shortcut | Action |
|----------|--------|
| `Space` | Pan mode toggle |
| `Delete` | Delete selected item |

---

## 14. Troubleshooting

### Common Issues

#### "No Images Loaded"

- Make sure you've imported an image folder or video first
- Check that the folder contains supported image formats

#### Fusion Results Look Blurry or Ghosted

- Enable image registration (ECC and/or Homography)
- If the subject grows or shrinks across the stack, also enable **Scale (focus breathing)**
- Ensure images are properly aligned before fusion
- Try a different fusion algorithm
- Increase kernel size for smoother results

#### Bright Halo Around the Subject

- This is the defocused foreground's glow being copied from the background-focused frames
- On a **Depth Map** method, try this before touching **Halo suppression** — the focus measure discounts a glow on its own, and the dial fills the ring with defocused pixels rather than recovering what is behind it
- If a glow still survives, raise **Halo suppression** to roughly the halo's width in pixels
- Larger radii trade away genuine background detail near the subject's outline, so use the smallest value that clears the glow

#### Flat, Washed-Out Ring Tracing the Subject's Outline

- Distinct from the glow above: this is a *loss* of background detail in a band around the subject, not a bright fringe, and it grows with the kernel size
- Fixed in 1.33.0 — if you are seeing it, you are on an older build
- If it persists, check that **Halo suppression** is 0; a radius comparable to the kernel size forces this band by design

#### Blotchy, Torn-Looking Background Behind the Subject

- Distinct from both entries above: a mosaic of hard-edged patches, each a slightly different shade or blur, over an area that is out of focus in every frame — a background well behind the focus sweep, or any dark, flat patch
- The cause is that no frame is genuinely sharper than the others there, so per-pixel selection has nothing to go on and neighbouring pixels take frames from opposite ends of the stack
- Fixed in 1.34.0 — if you are seeing it, you are on an older build. Depth maps are now despeckled before the pixels are gathered, so such an area comes out as one coherent choice
- A *smooth* shading variation over that area is normal and is not this artifact; if you want it perfectly even, a transform-domain method (DTCWT, Pyramid) blends the defocused background rather than selecting from it

#### StackMFF-V4 Unavailable

- Install PyTorch: `pip install torch torchvision`
- For GPU support, install CUDA-enabled PyTorch
- Check via **Help → Environment Info**

#### Out of Memory Errors

- Enable tile processing in **Settings → Tile**
- Reduce tile block size
- Use smaller kernel sizes
- Resize images to smaller dimensions

#### Registration Fails

- Try only one registration method at a time
- Reduce the downscale width in **Settings → Registration**
- Ensure images have sufficient contrast and features

#### Slow Processing

- Increase thread count in **Settings → Thread Count Settings**
- Use a more powerful CPU
- Reduce image resolution
- Use simpler fusion algorithms (Guided Filter instead of StackMFF-V4)
- Click **Stop** to interrupt a render that is taking longer than expected, then adjust settings and try again

### Getting Help

1. **Environment Info**: Go to **Help → Environment Info** to check your installation
2. **Contact Support**: Go to **Help → Contact Us** for support information

---

## Appendix: Fusion Algorithm Details

### Guided Filter

The guided filter performs edge-preserving smoothing using a guidance image. It computes local linear transformations between the guidance and input images, then applies these to produce the output. This method excels at preserving edges while reducing noise.

### DCT (Discrete Cosine Transform)

DCT transforms images to the frequency domain where fusion decisions are made based on coefficient magnitudes. High-frequency components (details) are preserved while low-frequency components (smooth regions) are merged based on local energy.

### DTCWT (Dual-Tree Complex Wavelet Transform)

DTCWT provides multi-scale, multi-directional decomposition of images. It captures features at different scales and orientations, allowing for sophisticated fusion rules that preserve both texture and structure information.

### GFG-FGF

This algorithm uses a Generalized Four-neighborhood Gaussian approach to measure local focus, combined with Fast Guided Filter for weight map refinement. It efficiently identifies and merges in-focus regions.

### Pyramid

Each frame is decomposed into a Laplacian pyramid — a series of band-pass detail levels plus a low-frequency base. Every detail band is then a weighted mean across the stack, each frame weighted by how far its pooled local energy falls behind the best on offer: a frame twice behind the winner contributes well under a percent, so a real focus decision is still a decision, while frames that tie — a defocused background, where no frame resolves anything — are averaged instead of picked between. Energies are compared in units of each frame's own grain, so a bright noisy frame cannot win the areas that hold no detail. The low-frequency base is weighted by each frame's aggregate detail activity, so the frames that carry the detail carry the base with it. Collapsing the fused pyramid reconstructs the all-in-focus image, and every pixel is finally held inside the range its own frames span, so bands taken from different frames cannot add up to a value no frame had. Based on Burt and Adelson's Laplacian pyramid (1983); the textbook choose-max rule it starts from is still available as *Selectivity → Winner takes all*.

### Depth Map

A local Laplacian-energy focus measure is computed for every frame and pooled over an adjustable window. In **Max** mode each pixel is taken whole from the frame whose measure is highest there — an order-independent per-pixel argmax that produces an explicit depth map, so colour and noise never blend across a slice boundary. In **Average** mode the frames are blended in proportion to that measure with a small baseline weight, so a flat region — where every frame is equally (de)focused — collapses to the plain mean and recovers the stack's multi-frame signal-to-noise (a √N noise reduction), while textured regions are still dominated by the frame that actually holds the detail. This is the depth-map family used by tools such as Zerene DMap and Helicon Methods A/B.

### StackMFF-V4

A deep learning approach using a neural network trained on large datasets of multi-focus image pairs. The network learns optimal fusion strategies automatically, providing state-of-the-art results when properly configured.

---

*OpenFocus is an open-source project. For updates, bug reports, and feature requests, please visit the project repository.*
