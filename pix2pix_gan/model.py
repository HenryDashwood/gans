import torch
from torch import nn


def ConvBlock(in_filters, out_filters, kernel, batch_norm, padding):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=kernel,
                bias=False,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_filters),
            nn.LeakyReLU(0.2),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=kernel,
                bias=False,
                stride=2,
                padding=padding,
            ),
            nn.LeakyReLU(0.2),
        )


def ConvTransBlock(in_filters, out_filters, kernel, dropout, padding):
    if dropout:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=kernel,
                bias=False,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_filters),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=kernel,
                bias=False,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_filters),
            nn.LeakyReLU(0.2),
        )


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.unet_down = nn.Sequential(
            ConvBlock(3, 64, 4, False, 1),  # (bs, 128, 128, 64)
            ConvBlock(64, 128, 4, True, 1),  # (bs, 64, 64, 128)
            ConvBlock(128, 256, 4, True, 1),  # (bs, 32, 32, 256)
            ConvBlock(256, 512, 4, True, 1),  # (bs, 16, 16, 512)
            ConvBlock(512, 512, 4, True, 1),  # (bs, 8, 8, 512)
            ConvBlock(512, 512, 4, True, 1),  # (bs, 4, 4, 512)
            ConvBlock(512, 512, 4, True, 1),  # (bs, 2, 2, 512)
            ConvBlock(512, 512, 4, True, 1),  # (bs, 1, 1, 512)
        )
        self.unet_up = nn.Sequential(
            ConvTransBlock(512, 512, 4, True, 1),  # (bs, 2, 2, 1024)
            ConvTransBlock(1024, 512, 4, True, 1),  # (bs, 4, 4, 1024)
            ConvTransBlock(1024, 512, 4, True, 1),  # (bs, 8, 8, 1024)
            ConvTransBlock(1024, 512, 4, False, 13),  # (bs, 16, 16, 1024)
            ConvTransBlock(1024, 256, 4, False, 25),  # (bs, 32, 32, 512)
            ConvTransBlock(512, 128, 4, False, 49),  # (bs, 64, 64, 256)
            ConvTransBlock(256, 64, 4, False, 97),  # (bs, 128, 128, 128)
        )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, inputs):
        skips = []
        for layer in self.unet_down:
            outputs = layer(inputs)
            skips.append(outputs)
            inputs = outputs
        rev_skips = reversed(skips[:-1])  # down has one more layer than up
        for layer, skip in zip(self.unet_up, rev_skips):
            outputs = layer(inputs)
            inputs = torch.cat((outputs, skip), 1)
        return self.last(inputs)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(6, 64, 4, False, 1),  # (bs, 128, 128, 64)
            ConvBlock(64, 128, 4, True, 1),  # (bs, 64, 64, 128)
            ConvBlock(128, 256, 4, True, 1),  # (bs, 32, 32, 256)
            nn.ZeroPad2d(padding=1),  # (bs, 34, 34, 256)
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=1, bias=False
            ),  # (bs, 31, 31, 512)
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2),
            nn.ZeroPad2d(padding=1),  # (bs, 33, 33, 512)
            nn.Conv2d(
                in_channels=512, out_channels=1, kernel_size=4, stride=1, bias=False
            ),  # (bs, 30, 30, 1)
            torch.nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.net(inputs)
