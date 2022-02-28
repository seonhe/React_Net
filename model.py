import torch
import torch.nn as nn
import torch.nn.functional as F
from react import Sign, RSign, RPReLU, GeneralConv2d, ReactBase

def conv3x3(in_channel, out_channel, stride=1,padding=1):
    return nn.Conv2d(in_channels=in_channel, out_channels= out_channel, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channels=in_channel, out_channels= out_channel, kernel_size=1, stride=stride, bias=False)

# Normal Blocks for baseline model
class Base_Normal_Block(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.layer1 = nn.Sequential(
            Sign(in_channel),
            conv3x3(in_channel, in_channel, stride=1, padding=1),
            nn.BatchNorm2d(in_channel)
        )

        self.layer2 = nn.Sequential(
            Sign(in_channel),
            conv1x1(in_channel, in_channel, stride=1),
            nn.BatchNorm2d(in_channel)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out + x
        out = self.layer2(out)
        return out + x

# Reduction Blocks for baseline model
class Base_Reduction_Block(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.layer1 = nn.Sequential(
                Sign(in_channel),
                conv3x3(in_channel,in_channel, stride=2,padding=1),
                nn.BatchNorm2d(in_channel)
        )

        self.layer2_1 = nn.Sequential(
                Sign(in_channel),
                conv1x1(in_channel,in_channel,stride=1),
                nn.BatchNorm2d(in_channel)
        )

        self.layer2_2 = nn.Sequential(
                Sign(in_channel),
                conv1x1(in_channel,in_channel,stride=1),
                nn.BatchNorm2d(in_channel)
        )

        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)        

    def forward(self, x):
        out1 = self.layer1(x)
        pooled_x = self.pool(x)
        out1 = out1 + pooled_x

        out2_1 = self.layer2_1(out1)
        out2_1 = out2_1 + pooled_x

        out2_2 = self.layer2_2(out1)
        out2_2 = out2_2 + pooled_x

        return torch.cat([out2_1, out2_1], dim=1)

# 3 Normal, 3 Reduction
class BaselineModel(ReactBase):
    def __init__(self, channel_array):
        super().__init__()
        
        self.channels = channel_array
        self.blocks = []
        for i, channel in enumerate(self.channels):
            if i == 0:
                self.blocks.append(Base_Normal_Block(channel))
            elif i == i-1:
                self.blocks.append(Base_Normal_Block(channel))
            else:
                self.blocks.append(Base_Reduction_Block(channel))

        self.base_block = nn.ModuleList(self.blocks)

    def forward(self, x):
        y = self.base_block(x)
        return F.log_softmax(y.squeeze(dim=2).squeeze(dim=2), dim=1)



#######################################################################################################

# Normal Blocks for ReActNet
class React_Normal_Block(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.layer1 = nn.Sequential(
            RSign(in_channel),
            GeneralConv2d(in_channel, in_channel, "scaled_sign", kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channel)
        )

        self.layer2 = nn.Sequential(
            RSign(in_channel),
            GeneralConv2d(in_channel, in_channel, "scaled_sign", kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_channel)
        )

        self.rprelu = RPReLU(in_channel)


    def forward(self, x):
        out = self.layer1(x)
        out = out + x

        out_r = self.rprelu(out)

        out = self.layer2(out_r)
        out = out + out_r

        out = self.rprelu(out)

        return out

# Reduction Blocks for ReActNet
class React_Reduction_Block(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.layer1 = nn.Sequential(
                RSign(in_channel),
                GeneralConv2d(in_channels=in_channel, out_channels=in_channel, conv="scaled_sign",kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(in_channel)
        )

        self.layer2 = nn.Sequential(
                RSign(in_channel),
                GeneralConv2d(in_channels=in_channel, out_channels=in_channel, conv="scaled_sign", kernel_size=1, stride=1, padding=1),
                nn.BatchNorm2d(in_channel)
        )

        self.rprelu = RPReLU(in_channel)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.layer1(x)
        pooled_x = self.pool(x)
        out1 = out1 + pooled_x
        out1 = self.rprelu(out1)

        out2_1 = self.layer2(out1)
        out2_2 = self.layer2(out1)

        out2_1 = out2_1 + out1
        out2_2 = out2_2 + out1

        return torch.cat([out2_1, out2_2],dim=1)



class ReactModel(ReactBase):
    def __init__(self, channel_array):
        super().__init__()
        
        self.channels = channel_array
        self.blocks = []
        for i, channel in enumerate(self.channels):
            if i == 0:
                self.blocks.append(React_Normal_Block(channel))
            elif i == i-1:
                self.blocks.append(React_Normal_Block(channel))
            else:
                self.blocks.append(React_Reduction_Block(channel))

        self.base_block = nn.ModuleList(self.blocks)

    def forward(self, x):
        y = self.base_block(x)
        return F.log_softmax(y.squeeze(dim=2).squeeze(dim=2), dim=1)








