import math
import torch
from torch import nn


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class WScaleLayer(nn.Module):
    def __init__(self, incoming, gain=2):
        super(WScaleLayer, self).__init__()
        self.gain = gain
        self.scale = (self.gain / incoming.weight[0].numel()) ** 0.5

    def forward(self, input):
        return input * self.scale


def GConvBlock(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        PixelNormLayer(),
    )


class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int, nc: int, image_size: int, max_levels: int):
        super(Generator, self).__init__()
        self.max_levels = max_levels

        self.blocks = nn.ModuleList()
        self.toRGBs = nn.ModuleList()
        self.blocks.append(
            nn.Sequential(
                GConvBlock(
                    in_channels=nz,
                    out_channels=ngf * 48,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                GConvBlock(
                    in_channels=ngf * 48,
                    out_channels=ngf * 32,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )
        )

        i, j = 5, 4
        while j <= image_size:
            in_channels = int(ngf * (2 ** i))
            out_channels = int(ngf * (2 ** (i - 1)))
            self.blocks.append(
                nn.Sequential(
                    GConvBlock(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    GConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                )
            )
            self.toRGBs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=nc,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.Tanh(),
                )
            )

            i -= 1
            j *= 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, level):
        alpha = False
        level = min(int(math.ceil(level)), self.max_levels)
        feature_map = self.blocks[0](inputs)
        for i in range(1, level):
            feature_map = self.blocks[i](feature_map)
            if level > 1 and i == level - 1 and alpha != 0:
                prev_output = nn.functional.upsample(feature_map, scale_factor=2)
                prev_output = self.toRGBs[i](feature_map)
        output = self.toRGBs[level - 1](feature_map)
        if alpha != 0:
            return output * alpha + prev_output * (1 - alpha)
        else:
            return output


def DConvBlock(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(self, ndf: int, nc: int, image_size: int, max_levels: int):
        super(Discriminator, self).__init__()
        self.max_levels = max_levels
        self.fromRGBs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        i, j = 0, 4
        while j <= image_size:
            in_channels = int(ndf * (2 ** i))  # 1 => 2 => 4 => 8  => 16
            out_channels = int(ndf * (2 ** (i + 1)))  # 2 => 4 => 8 => 16 => 32
            self.fromRGBs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=nc,
                        out_channels=in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
            self.blocks.append(
                nn.Sequential(
                    DConvBlock(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    DConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                )
            )
            i += 1
            j *= 2

        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=ndf * (2 ** (i + 1)),
                    out_channels=1,
                    kernel_size=2,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Sigmoid(),
            )
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, level):
        alpha = False
        level = min(int(math.ceil(level)), self.max_levels)
        feature_map = self.fromRGBs[self.max_levels - level](inputs)
        feature_map = self.blocks[self.max_levels - level](feature_map)
        if level > 1 and alpha != 0:
            prev_input = nn.functional.avg_pool2d(inputs, kernel_size=2, stride=2)
            prev_feature_map = self.fromRGBs[self.max_levels - level + 1](prev_input)
            feature_map = alpha * feature_map + (1 - alpha) * prev_feature_map

        for i in range(self.max_levels - level + 1, self.max_levels):
            feature_map = self.blocks[i](feature_map)
        stdv = torch.std(feature_map, dim=0)
        y = torch.cat((feature_map, stdv.unsqueeze(0).expand_as(feature_map)), dim=1)
        ret = self.blocks[-1](y).view(-1, 1)
        return ret
