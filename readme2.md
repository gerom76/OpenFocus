# Enabling CUDA GPU Acceleration in OpenFocus

OpenFocus can use an NVIDIA GPU to accelerate the **StackMFF-V4** (AI) fusion method. The other fusion methods — Guided Filter, DCT, DTCWT, and GFG-FGF — always run on the CPU, so a GPU only speeds up StackMFF-V4.

The app detects the GPU automatically through PyTorch. If the status bar shows **GPU: N/A**, it means the installed PyTorch is the CPU-only build — not that anything is wrong with your GPU or driver.

## Requirements

- An NVIDIA GPU (GTX 10-series or newer; RTX cards recommended)
- A reasonably recent NVIDIA driver (the PyTorch wheel bundles its own CUDA runtime, so you do **not** need to install the CUDA Toolkit separately)
- Python with `pip`

## Steps

### 1. Check what you currently have

```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

- If the version ends in `+cpu` (e.g. `2.13.0+cpu`) or `cuda: False` is printed, you have the CPU-only build — continue below.
- If it prints `cuda: True`, CUDA is already enabled and you are done.

### 2. Replace the CPU-only build with the CUDA build

```powershell
pip uninstall -y torch torchvision
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

Notes:

- The download is several GB; it may take a few minutes.
- `cu130` (CUDA 13.0) covers all current NVIDIA cards. For older setups you can use `cu126` instead.
- If a newer torch version is available, you can omit the version pins — just keep the `--index-url` so pip fetches the CUDA wheels rather than the default CPU wheels.
- If you run OpenFocus in a virtual environment, run these commands with that environment activated.

### 3. Verify

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output:

```text
True NVIDIA GeForce RTX 4080   (your GPU model)
```

### 4. Restart OpenFocus

- The status bar (bottom right) should now show **GPU: CUDA** instead of **GPU: N/A**.
- Select **StackMFF-V4** in the Fusion panel to actually use the GPU.
- The console prints `Running AI fusion on CUDA...` when rendering.

## Troubleshooting

- **`torch.OutOfMemoryError: CUDA out of memory` during StackMFF-V4 rendering** — the AI model's memory use grows with tile area and with the square of the number of images in the stack, and large renders can exceed the GPU's VRAM. The app automatically retries with smaller batches and falls back to CPU if a single tile still does not fit, but to keep everything on the GPU, lower **Tile Block Size** in Settings (e.g. 512 instead of 1024) — this quarters the memory per tile — and/or reduce the number of images per stack.
- **Still shows GPU: N/A after reinstalling** — make sure the `pip`/`python` you used is the same interpreter that launches OpenFocus (`(Get-Command python).Source` shows which one is on your PATH).
- **Status bar shows GPU: Err** — PyTorch imported but GPU probing failed; update your NVIDIA driver.
- **`nvidia-smi` not found or fails** — the NVIDIA driver is missing or broken; reinstall the driver first.
- **macOS** — Apple Silicon GPUs are supported automatically via MPS (`pip install torch` is enough); the status bar shows **GPU: MPS**.
