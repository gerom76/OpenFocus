# Weights

`*.pth` is gitignored, so a fresh checkout starts without either file.

| File             | Used by                   | Source                  |
|------------------|---------------------------|-------------------------|
| `stackmffv4.pth` | StackMFF-V4 fusion method | ships with the project  |
| `ifcnn.pth`      | IFCNN refinement stage    | fetch it (see below)    |

## ifcnn.pth

The IFCNN refinement stage stays disabled (greyed out, with the reason in its
tooltip) until this file exists. `IFCNN-MAX` is the variant intended for
multi-focus input and the one matching the default fusion scheme in
[`core/models/ifcnn_network.py`](../core/models/ifcnn_network.py):

```bash
curl -sSL https://raw.githubusercontent.com/uzeful/IFCNN/master/Code/snapshots/IFCNN-MAX.pth \
     -o weights/ifcnn.pth
```

- 337,094 bytes, 83,843 parameters
- sha256 `6022b44c88a2f0367ba9b08fcdf0d663745d9a966fca0ee1f41867797b6ff53f`

The checkpoint carries every layer, including the frozen 7x7 ResNet101 stem, so
nothing is downloaded at runtime. Checkpoints with or without convolution biases
both load — the layer layout is derived from the file — and the pre-buffer
checkpoints that lack `num_batches_tracked` are accepted, since those counters
only steer momentum while training. Any other mismatch is rejected outright
rather than loading a half-initialized network.
