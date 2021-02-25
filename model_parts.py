import torch.nn as nn
import torch


def DoubleConv(in_channel, out_channel):
    # Does not change the size
    return nn.Sequential(nn.Conv3d(in_channel, out_channel, 3, padding=1),
                         nn.BatchNorm3d(num_features=out_channel),
                         nn.ReLU(),
                         nn.Conv3d(out_channel, out_channel, 3, padding=1),
                         nn.BatchNorm3d(num_features=out_channel),
                         nn.ReLU())


class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()
        self.conv = DoubleConv(in_channel, out_channel)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        conv = self.conv(x)
        out = self.maxpool(conv)
        return conv, out


class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel, up_scale=True):
        """
        :param in_channel:
        :param out_channel:
        :param up_scale: do up sampling or not.
        """
        super(UpConv, self).__init__()
        self.up_scale = up_scale
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dobuleconv = DoubleConv(in_channel, out_channel)

    def forward(self, x, conv=None):
        if self.up_scale:
            x = self.up(x)

        x = torch.cat([x, conv], dim=1)
        x = self.dobuleconv(x)
        return x
