# PyInstaller Build Commands

This document converts the examples in `build_command.txt` to a Markdown reference. Each command is written for PowerShell (Windows) and uses PyInstaller to create a packaged application.

---

## 1) Build without `torch` (exclude torch)

Use this when you don't want to bundle PyTorch into the installer. Produces a single-file executable (`--onefile`) without console (`--noconsole`).

```powershell
pyinstaller --clean --noconfirm --onefile --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-all PyQt6 `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --exclude-module torch `
  --exclude-module torchvision `
  --exclude-module cupy `
  --exclude-module cupyx `
  --exclude-module cupy_backends `
  main.py
```

Notes:
- `--exclude-module` prevents `torch` / `torchvision` from being scanned and bundled.
- Use when target machines do not need GPU/torch features.
- `--add-data "ui;ui"` includes UI styles and resources.
- `--add-data "docs;docs"` includes documentation files.
- The CuPy excludes are **not optional** — see the warning below. Since 1.30.5
  nothing in the app imports CuPy at all (registration no longer warps on the
  GPU), so every build here excludes it; on this one it also prevents the failure
  mode described next.

### ⚠ Why CuPy must be excluded from the no-torch build

CuPy's extension modules link against the CUDA DLLs that ship inside
`site-packages/torch/lib` (`cublasLt64_13.dll`, `cufft64_12.dll`, ...). Even with
`--exclude-module torch`, PyInstaller's dependency scan copies those DLLs into a
`torch/lib/` folder inside the bundle, which has two consequences:

1. **~1.2 GB of dead weight.** Windows does not search bundle subdirectories for
   DLLs, so CuPy could not load them anyway - the frozen app used to log
   `[Info] Cupy not found. Using CPU for warping.` and carry on. Since 1.30.5 it
   does not even look: there is no CuPy code path left to fall back from.
2. **A fake `torch` package.** `torch/` has no `__init__.py`, so `import torch`
   succeeds and returns an *empty* implicit namespace package. Every
   `except ImportError` CPU fallback is bypassed and the first attribute access
   fails with `AttributeError: module 'torch' has no attribute 'cuda'` — during
   rendering, after the images are already loaded and registered.

`utils/torch_env.py` detects that stub at startup and makes the import fail
properly, so a bundle built without the CuPy excludes still falls back to CPU
instead of crashing. Excluding CuPy removes the cause (and the 1.2 GB).

If a build must keep CuPy anyway — for something outside this application, since
nothing inside it uses CuPy any more — strip the leftovers in a `.spec` file
instead:

```python
a.binaries = [b for b in a.binaries if not b[0].lower().startswith('torch\\')]
```

---

## 2) Include `torch`, single-file (--onefile)

Bundle `torch` and `torchvision` into a single-file executable. This increases exe size and build time significantly.

```powershell
pyinstaller --clean --noconfirm --onefile --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-all PyQt6 `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all torch `
  --collect-all torchvision `
  main.py
```

Notes:
- Single-file with `torch` may hit antivirus false positives and will be large. Consider `--onedir` if size/time is an issue.
- Includes `ui` and `docs` directories for complete application functionality.

---

## 3) GPU build: `torch`, output as one directory (`--onedir`) ⭐

The recommended GPU build. `torch` drives GPU fusion (DTCWT / GFF / DCT / pyramid /
GFG-FGF / StackMFF-V4) and the nvJPEG decode path.

```powershell
pyinstaller --clean --noconfirm --onedir --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-all PyQt6 `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all torch `
  --collect-all torchvision `
  --collect-data pytorch_wavelets `
  --exclude-module cupy `
  --exclude-module cupyx `
  --exclude-module cupy_backends `
  main.py
```

Notes:
- `--onedir` is strongly preferred here: a `--onefile` GPU build extracts several GB
  to `%TEMP%` on *every* launch, which costs a minute of start-up and the same amount
  of free disk. Zip `dist\OpenFocus\` if you need a single file to hand over.
- **CuPy is excluded from this build too, as of 1.30.5.** It used to be collected
  here because registration warped on the GPU through
  `cupyx.scipy.ndimage.map_coordinates`. That path is deleted - it resampled
  bilinearly, which cost 44% of the high-frequency energy the focus measure reads,
  and at the spline order that matches the CPU's Lanczos4 it was the slower of the
  two (`docs/REGISTRATION_IMPROVEMENTS.md` item 4). Nothing in the app imports
  CuPy now, so collecting it only adds ~1.2 GB. The `--collect-all cuda` and
  `--hidden-import graphlib` flags that CuPy needed inside a bundle go with it.
- `--collect-data pytorch_wavelets` ships the `.npz` wavelet coefficients that the GPU
  DTCWT path loads at runtime (via the `pkg_resources` shim in
  `fusion_methods/dtcwt_torch.py`). Without them GPU DTCWT falls back to the CPU.
- A CUDA toolkit on the *target* machine is no longer needed for anything: it was
  required only because CuPy compiles its kernels at runtime with NVRTC. GPU fusion
  via `torch` never needed one — torch ships its own CUDA runtime in `torch\lib`.
- Includes `ui` and `docs` directories for complete application functionality.

---

## Common options explained
- `--clean`: Clean PyInstaller cache and temporary files before building.
- `--noconfirm`: Overwrite output directory without asking.
- `--onefile` / `--onedir`: Bundle into single executable or directory.
- `--noconsole`: Hide console window (useful for GUI apps).
- `--icon`: App icon file path.
- `--add-data "src;dest"`: Include extra data files; format on Windows is `"src;dest"` (note backslashes in paths).
  - Required directories: `assets`, `weights`, `ui`, `docs`
- `--collect-all <package>`: Collect package data, binaries, submodules for the named package.
- `--copy-metadata <package>`: Copy package metadata (useful for packages like `imageio`).
- `--exclude-module <module>`: Prevent a specific module from being bundled.

## ⭐ Required Resource Directories

All PyInstaller commands MUST include these directories:

| Directory | Purpose | Required |
|-----------|---------|----------|
| `assets` | Icons, images, UI resources | ✅ Yes |
| `weights` | AI model files (StackMFF-V4) | ✅ Yes |
| `ui` | UI styles and resources (styles.py) | ✅ Yes |
| `docs` | Documentation files | ✅ Yes |

**Failure to include all directories will result in `FileNotFoundError` at runtime.**

---

## 4) Build with Cross-Platform Drag-and-Drop Support

Use the spec file (`docs/OpenFocus.spec`) for proper macOS app bundle and drag-and-drop support:

```powershell
# Windows (PowerShell)
pyinstaller --clean --noconfirm --onefile --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-all PyQt6 `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all torch `
  --collect-all torchvision `
  main.py
```

```bash
# macOS (bash) - Creates .app bundle with dock drag-and-drop support
pyinstaller --clean --noconfirm --onedir --noconsole \
  --name OpenFocus \
  --icon "./assets/OpenFocus.icns" \
  --add-data "assets:assets" \
  --add-data "weights:weights" \
  --add-data "ui:ui" \
  --add-data "docs:docs" \
  --collect-all PyQt6 \
  --collect-all scipy \
  --copy-metadata imageio \
  --collect-data dtcwt \
  --collect-all torch \
  --collect-all torchvision \
  --osx-bundle-identifier com.openfocus.app \
  main.py
```

```bash
# Linux (bash) - With desktop file integration
pyinstaller --clean --noconfirm --onedir --noconsole \
  --name openfocus \
  --icon "./assets/OpenFocus.png" \
  --add-data "assets:assets" \
  --add-data "weights:weights" \
  --add-data "ui:ui" \
  --add-data "docs:docs" \
  --collect-all PyQt6 \
  --collect-all scipy \
  --copy-metadata imageio \
  --collect-data dtcwt \
  --collect-all torch \
  --collect-all torchvision \
  main.py

# Install desktop file for taskbar drag-and-drop support
cp assets/openfocus.desktop ~/.local/share/applications/
update-desktop-database ~/.local/share/applications/
```

### macOS Requirements for Dock Drag-and-Drop

The `assets/Info.plist` file defines document types for drag-and-drop. Copy it to the app bundle:

```bash
# After building, copy Info.plist to app bundle
cp assets/Info.plist OpenFocus.app/Contents/Info.plist
```

### Linux Desktop File Installation

For full taskbar/dock drag-and-drop support on Linux:

```bash
# System-wide installation (requires root)
sudo cp assets/openfocus.desktop /usr/share/applications/
sudo cp assets/OpenFocus.png /usr/share/pixmaps/openfocus.png
sudo update-desktop-database /usr/share/applications/

# Or user-specific installation
mkdir -p ~/.local/share/applications/
cp assets/openfocus.desktop ~/.local/share/applications/
cp assets/OpenFocus.png ~/.local/share/pixmaps/
update-desktop-database ~/.local/share/applications/
```

---

## Drag-and-Drop Feature Support by Platform

| Platform | Taskbar/Dock Icon | Window | EXE/Shortcut |
|----------|-------------------|--------|--------------|
| macOS | ✅ Full support | ✅ Full support | N/A |
| Linux | ⚠️ Partial (DE-dependent) | ✅ Full support | N/A |
| Windows | ⚠️ Limited | ✅ Full support | ✅ Full support |

### Platform Notes

- **macOS**: Dock drag-and-drop uses `QFileOpenEvent` via Info.plist
- **Linux**: Desktop file with `%U` argument handles file URLs
- **Windows**: Drag-to-EXE/shortcut passes files via command-line arguments

---
