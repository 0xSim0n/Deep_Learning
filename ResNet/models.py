import torch
from torch import nn
from torch.nn import functional as F


MODEL_TYPES = {"resnet", "plain", "highway"}


class ShortcutA(nn.Module):
    """Parameter-free shortcut (option A): subsample by stride and zero-pad the new channels."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.pad = out_channels - in_channels

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, model_type="resnet", gate_bias=-2.0):
        super().__init__()
        self.model_type = model_type
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutA(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        if model_type == "highway":
            self.gate = nn.Conv2d(out_channels, out_channels, kernel_size=1)
            self.gate_bias = gate_bias
        self.last_gate_mean = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.model_type == "resnet":
            out = out + self.shortcut(x)
        elif self.model_type == "highway":
            carry = self.shortcut(x)
            t = torch.sigmoid(self.gate(carry))
            self.last_gate_mean = t.detach().mean()
            out = t * out + (1.0 - t) * carry
        return F.relu(out)


class CIFARResNet(nn.Module):
    """ResNet for CIFAR-10 (He et al. 2015, Sec. 4.2): 6n + 2 layers, feature maps 16/32/64."""

    def __init__(self, model_type="resnet", depth=56):
        super().__init__()
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {sorted(MODEL_TYPES)}")
        if (depth - 2) % 6 != 0:
            raise ValueError("depth must be 6n + 2 (e.g. 20, 32, 44, 56, 110)")

        n = (depth - 2) // 6

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.stage1 = self._make_stage(16, 16, n, 1, model_type)
        self.stage2 = self._make_stage(16, 32, n, 2, model_type)
        self.stage3 = self._make_stage(32, 64, n, 2, model_type)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for block in self.highway_blocks():
            nn.init.constant_(block.gate.bias, block.gate_bias)

    @staticmethod
    def _make_stage(in_channels, out_channels, blocks, stride, model_type):
        layers = [BasicBlock(in_channels, out_channels, stride, model_type)]
        layers += [BasicBlock(out_channels, out_channels, 1, model_type) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def highway_blocks(self):
        return [m for m in self.modules() if isinstance(m, BasicBlock) and m.model_type == "highway"]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)
