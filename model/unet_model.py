""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

from model.unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = trilinear

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, trilinear)
        self.up2 = Up(64, 32, trilinear)
        self.up3 = Up(32, 16, trilinear)
        self.up4 = Up(16, 8, trilinear)
        self.outc = OutConv(8, n_classes, 8)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

if __name__ == '__main__':
    net = UNet(n_channels=32, n_classes=1)
    d = torch.rand(1, 32, 384, 384)
    result = net(d)
    print(result.shape)