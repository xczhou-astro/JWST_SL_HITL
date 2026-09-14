import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Basic residual block with optional Bayesian dropout."""

    def __init__(self, in_channels, out_channels, stride=1, bayesian=False, dropout_rate=0.1):
        super().__init__()
        self.bayesian = bayesian

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        if self.bayesian:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.bayesian:
            out = self.dropout(out)

        out = out + residual
        out = self.act(out)
        return out


class ResNetRGBClassifier(nn.Module):
    """
    Small ResNet-like classifier for RGB 128x128 images.
    Input:  [B, 3, 128, 128]
    Output: [B] (logits)
    """

    def __init__(
        self,
        num_classes=1,
        bayesian=False,
        dropout_rate=0.2,
        base_channels=32,
        blocks_per_stage=(2, 2, 2),
    ):
        super().__init__()
        self.bayesian = bayesian

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(
            base_channels,
            base_channels,
            blocks_per_stage[0],
            stride=1,
            bayesian=bayesian,
            dropout_rate=dropout_rate,
        )
        self.stage2 = self._make_stage(
            base_channels,
            base_channels * 2,
            blocks_per_stage[1],
            stride=2,
            bayesian=bayesian,
            dropout_rate=dropout_rate,
        )
        self.stage3 = self._make_stage(
            base_channels * 2,
            base_channels * 4,
            blocks_per_stage[2],
            stride=2,
            bayesian=bayesian,
            dropout_rate=dropout_rate,
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(base_channels * 4, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, bayesian, dropout_rate):
        blocks = [
            ResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                bayesian=bayesian,
                dropout_rate=dropout_rate,
            )
        ]
        for _ in range(num_blocks - 1):
            blocks.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    bayesian=bayesian,
                    dropout_rate=dropout_rate,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        if self.bayesian:
            x = self.head_dropout(x)

        logits = self.classifier(x)
        return logits.squeeze(-1)

    def enable_mc_dropout(self):
        """Enable dropout layers at inference for Monte Carlo sampling."""
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
