import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, block_1_channels, block_2_channels, block_3_channels, block_4_channels, bottleneck_channels):
        super().__init__()
        self.enc1 = self.encoder_block(1, block_1_channels)
        self.enc2 = self.encoder_block(block_1_channels, block_2_channels)
        self.enc3 = self.encoder_block(block_2_channels, block_3_channels)
        self.enc4 = self.encoder_block(block_3_channels, block_4_channels)
        self.bottleneck = self.encoder_block(block_4_channels, bottleneck_channels)
        
        self.up4 = self.decoder_block(bottleneck_channels, block_4_channels)
        self.up3 = self.decoder_block(block_4_channels*2, block_3_channels)
        self.up2 = self.decoder_block(block_3_channels*2, block_2_channels)
        self.up1 = self.decoder_block(block_2_channels*2, block_1_channels)

        self.final_conv = nn.Conv2d(block_1_channels*2, 3, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.up4(b)
        d4 = torch.cat((d4, e4), dim=1)

        d3 = self.up3(d4)
        d3 = torch.cat((d3, e3), dim=1)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)

        return torch.tanh(self.final_conv(d1))