"""IFCNN network definition.

Reference:
    Y. Zhang, Y. Liu, P. Sun, H. Yan, X. Zhao, L. Zhang,
    "IFCNN: A general image fusion framework based on convolutional neural
    network", Information Fusion, vol. 54, pp. 99-118, 2020.

Layer names match the official release (https://github.com/uzeful/IFCNN) so the
published IFCNN-MAX / IFCNN-SUM / IFCNN-MEAN checkpoints load without
conversion. conv1 is the frozen 7x7 stem taken from a pretrained ResNet101; its
weights live in the checkpoint, so no torchvision download is needed here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature fusion schemes (same numbering as the official implementation)
FUSE_MAX = 0
FUSE_SUM = 1
FUSE_MEAN = 2


class ConvBlock(nn.Module):
    """3x3 convolution + batch norm + ReLU, using replicate padding."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, (1, 1, 1, 1), mode='replicate')
        out = self.conv(out)
        out = self.bn(out)
        return self.relu(out)


class IFCNN(nn.Module):
    """Fully convolutional fusion network: encode each input, merge the feature
    maps element-wise, then reconstruct a single image.

    Being fully convolutional (no down-sampling), it accepts any input size, so
    tiles can be processed independently.
    """

    def __init__(self, fuse_scheme: int = FUSE_MAX, conv1_bias: bool = False,
                 conv2_bias: bool = False, conv3_bias: bool = False, conv4_bias: bool = False):
        super().__init__()
        self.fuse_scheme = fuse_scheme
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0, bias=conv1_bias)
        self.conv2 = ConvBlock(64, 64, bias=conv2_bias)
        self.conv3 = ConvBlock(64, 64, bias=conv3_bias)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=conv4_bias)

    def encode(self, x):
        """Extract the 64-channel feature map of a single normalized image."""
        out = F.pad(x, (3, 3, 3, 3), mode='replicate')
        out = self.conv1(out)
        return self.conv2(out)

    def decode(self, features):
        """Reconstruct an image from a fused feature map."""
        return self.conv4(self.conv3(features))

    def accumulate(self, running, features):
        """Merge one more feature map into the running fusion accumulator.

        Accumulating incrementally keeps memory at two feature maps instead of
        one per input image, which matters for a full focus stack.
        """
        if running is None:
            return features
        if self.fuse_scheme == FUSE_MAX:
            return torch.max(running, features)
        return running + features  # SUM and MEAN share the accumulation step

    def finalize(self, running, count: int):
        """Turn the accumulator into the fused feature map."""
        if self.fuse_scheme == FUSE_MEAN and count > 0:
            return running / float(count)
        return running

    def forward(self, *tensors):
        running = None
        for tensor in tensors:
            running = self.accumulate(running, self.encode(tensor))
        return self.decode(self.finalize(running, len(tensors)))


def build_ifcnn_from_state_dict(state_dict, fuse_scheme: int = FUSE_MAX) -> IFCNN:
    """Build a network whose bias layout matches the checkpoint being loaded.

    Different IFCNN releases were trained with and without conv biases; deriving
    the flags from the checkpoint keeps ``load_state_dict(strict=True)`` usable.
    """
    return IFCNN(
        fuse_scheme=fuse_scheme,
        conv1_bias='conv1.bias' in state_dict,
        conv2_bias='conv2.conv.bias' in state_dict,
        conv3_bias='conv3.conv.bias' in state_dict,
        conv4_bias='conv4.bias' in state_dict,
    )
