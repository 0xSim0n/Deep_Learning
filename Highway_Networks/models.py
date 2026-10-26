import torch
from torch import nn
from torch.nn import functional as F


class HighwayBlock(nn.Module):
    def __init__(self, channels, gate_bias=-2.0):
        super().__init__()
        self.transform = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.kaiming_normal_(self.transform.weight, nonlinearity="relu")
        nn.init.zeros_(self.transform.bias)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.last_gate_mean = None

    def forward(self, x):
        h = F.relu(self.transform(x))
        t = torch.sigmoid(self.gate(x))
        self.last_gate_mean = t.detach().mean()
        return h * t + x * (1.0 - t)


class PlainBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return F.relu(self.conv(x))


class CIFAR10Net(nn.Module):
    def __init__(self, model_type="highway", blocks=10, channels=64):
        super().__init__()
        if model_type not in {"highway", "plain"}:
            raise ValueError("model_type must be 'highway' or 'plain'")
        if blocks < 3:
            raise ValueError("blocks must be at least 3")

        block_class = HighwayBlock if model_type == "highway" else PlainBlock
        counts = [blocks // 3 + (i < blocks % 3) for i in range(3)]

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        nn.init.kaiming_normal_(self.stem[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.stem[0].bias)
        self.stage1 = nn.Sequential(*(block_class(channels) for _ in range(counts[0])))
        self.stage2 = nn.Sequential(*(block_class(channels) for _ in range(counts[1])))
        self.stage3 = nn.Sequential(*(block_class(channels) for _ in range(counts[2])))
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, 10)

    def highway_blocks(self):
        return [m for m in self.modules() if isinstance(m, HighwayBlock)]

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.stage3(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)
