from model_parts import *
import torch.nn as nn

class UNet3D_4layer(nn.Module):
    def __init__(self, num_inputs):
        super(UNet3D_4layer, self).__init__()
        # num_inputs: number of input channels
        # min batch size depends on the depth of the model. Max depends on memory resources and batch size.
        self.acceptable_batch_sizes = [32, 48, 64, 80, 96, 112, 128]
        self.model_name = 'UNet3D_4layer'

        self.num_inputs = num_inputs

        # encoder
        self.down1 = DownConv(num_inputs, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 512)

        # middle
        self.mid = nn.Sequential(DoubleConv(512, 1024),
                                 nn.Conv3d(1024, 512, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))

        # decoder
        self.up1 = UpConv(512 + 512, 256, up_scale=False)
        self.up2 = UpConv(256 + 256, 128, up_scale=True)
        self.up3 = UpConv(128 + 128, 64, up_scale=True)
        self.up4 = UpConv(64 + 64, 1, up_scale=True)

    def forward(self, x):
        d1, x = self.down1(x)
        d2, x = self.down2(x)
        d3, x = self.down3(x)
        d4, x = self.down4(x)

        x = self.mid(x)

        x = self.up1(x, d4)
        x = self.up2(x, d3)
        x = self.up3(x, d2)
        x = self.up4(x, d1)

        return x


# a more shallow unet
class UNet3D_2layer(nn.Module):
    def __init__(self, num_inputs):
        super(UNet3D_2layer, self).__init__()
        # num_inputs: number of input channels
        # min batch size depends on the depth of the model.  Max depends on memory resources and batch size.
        self.acceptable_batch_sizes = [8, 16, 32, 48, 64, 80, 96, 112, 128]
        self.model_name = 'UNet3D_2layer'

        self.num_inputs = num_inputs

        # encoder
        self.down1 = DownConv(num_inputs, 64)
        self.down2 = DownConv(64, 128)

        # middle
        self.mid = nn.Sequential(DoubleConv(128, 256),
                                 nn.Conv3d(256, 128, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))

        # decoder
        self.up1 = UpConv(128 + 128, 64, up_scale=False)
        self.up2 = UpConv(64 + 64, 1, up_scale=True)

    def forward(self, x):
        d1, x = self.down1(x)
        d2, x = self.down2(x)

        x = self.mid(x)

        x = self.up1(x, d2)
        x = self.up2(x, d1)

        return x
