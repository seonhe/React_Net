import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


from react import RPReLU, RSign
from react import GeneralConv2d
from react import ReactBase
from react import firstconv3x3




# in_channels, out_channels, kernel_size, stride, padding, conv

# stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

#structure = [64] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2
#[64, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024 ]

class Model(ReactBase):
    def __init__(self, structure, **kwargs):
        super().__init__(structure, kwargs)
        self.blocks = nn.ModuleList()
        for i in range(len(structure)):
            if i == 0:
                self.blocks.append(firstconv3x3(in_channels=3, out_channels=structure[i],stride=2))
            elif structure[i] == structure[i-1]: # Normal block
                block = 'Normal'
                self.blocks.append(UpperBlock(in_channels=structure[i-1], out_channels=structure[i],block=block))
                self.blocks.append(LowerBlock(in_channels=structure[i-1],block=block))                      
        
            elif ((structure[i] == structure[i-1] * 2)): # Reduction block
                block = 'Reduction'
                self.blocks.append(UpperBlock(in_channels=structure[i-1], out_channels=structure[i], block=block))
                self.blocks.append(LowerBlock(in_channels=structure[i-1], block=block))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024,10)
        
    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class UpperBlock(nn.Module):
    def __init__(self,in_channels,out_channels, block): # block --> Reduction / Normal
        super().__init__()
        self.block = block
        if self.block == 'Reduction':
            self.pad = nn.ZeroPad2d(2)
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
            self.conv = GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv="scaled_sign",kernel_size=3,stride=2,padding=2)
        else: # self.block == 'Normal'
            self.conv = GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv="scaled_sign",kernel_size=3,stride=1,padding=1)

        #self.conv = GeneralConv2d(in_channels=in_channels, out_channels=out_channels, conv="scaled_sign",kernel_size=kernel_size,stride=stride,padding=padding)
        self.rsign = RSign(in_channels=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.rprelu = RPReLU(in_channels=in_channels)

    def forward(self,x):

        rsign_out = self.rsign(x)
        if self.block == 'Reduction':
            x = self.pad(x)
            x = self.pool(x)
        conv_out = self.conv(rsign_out)
        bn_out = self.bn(conv_out)
        shortcut_out = bn_out + x
        rprelu_out = self.rprelu(shortcut_out)

        return rprelu_out

class LowerBlock(nn.Module):
    def __init__(self,in_channels, block):
        super().__init__()
        self.block = block       

        self.rsign = RSign(in_channels=in_channels)
        self.conv = GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv="scaled_sign",kernel_size=1,stride=1,padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.rprelu = RPReLU(in_channels=in_channels)
        
        if self.block == 'Reduction':
            self.rsign2 = RSign(in_channels=in_channels)
            self.conv2 = GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv="scaled_sign",kernel_size=1,stride=1,padding=0)
            self.bn2 = nn.BatchNorm2d(in_channels)
            self.rprelu2 = RPReLU(in_channels=in_channels)

    def forward(self,x):

        rsign_out = self.rsign(x)
        conv_out = self.conv(rsign_out)
        bn_out = self.bn(conv_out)
        shortcut_out = bn_out + x
        out = self.rprelu(shortcut_out) # Normal block
        if self.block == 'Reduction':
            rsign2_out = self.rsign2(x)
            conv2_out = self.conv2(rsign2_out)
            bn2_out = self.bn2(conv2_out)
            shortcut2_out = bn2_out + x
            rprelu_out2 = self.rprelu(shortcut2_out)
            out = torch.cat([out, rprelu_out2], dim=1)

        return out


