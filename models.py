import torch.nn as nn
import torch

def DoubleConv(in_channel, out_channel):
    return nn.Sequential(nn.Conv3d(in_channel, out_channel, 3, padding=1),
                         nn.ReLU,
                         nn.Conv3d(out_channel, out_channel, 3, padding=1),
                         nn.ReLU)

class DownConv(nn.Module):
    # only down conv is implemented as a class so we can extract the features at conv output
    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()
        self. conv = DoubleConv(in_channel, out_channel)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        conv = self.conv(x)
        out = self.maxpool(conv)
        return conv, out


def UpConv(in_channel, out_channel):
    return nn.Sequential(DoubleConv(in_channel, out_channel),
                         nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True))


class FC_3D_UNet(nn.Module):
    def __init__(self):
        super(FC_3D_UNet, self).__init__()

        # encoder
        self.down1 = DownConv(1, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 512)

        # middle
        self.mid = nn.Sequential(DoubleConv(512, 1024),
                                 nn.Conv3d(1024, 512, 3, padding=1),
                                 nn.ReLU,
                                 nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True))

        # decoder
        self.up1 = UpConv(512 + 512, 256)
        self.up2 = UpConv(256 + 256, 128)
        self.up3 = UpConv(128 + 128, 64)
        self.up4 = UpConv(64 + 64, 1)


    def forward(self, x):
        d1, x = self.down1(x)
        d2, x = self.down2(x)
        d3, x = self.down3(x)
        d4, x = self.down4(x)

        x = self.mid(x)

        #TODO: check cat dim
        x = self.up1(torch.cat([x, d4], dim=1))
        x = self.up2(torch.cat([x, d3], dim=1))
        x = self.up3(torch.cat([x, d2], dim=1))
        x = self.up4(torch.cat([x, d1], dim=1))

        return x