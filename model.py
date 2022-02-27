import torch
import torch.nn as nn

from react import Sign, RSign, RPReLU

def conv3x3(in_channel, out_channel, stride=1,padding=1):
    return nn.Conv2d(in_channels=in_channel, out_channels= out_channel, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channels=in_channel, out_channels= out_channel, kernel_size=1, stride=stride, bias=False)

# Normal Blocks for baseline model
class Base_Normal_Block(nn.Module):
    def __init__(self, in_channel):
        self.layer1 = nn.Sequential([
            Sign(in_channel),
            conv3x3(in_channel, in_channel, stride=1, padding=1),
            nn.BatchNorm2d(in_channel)
        ])

        self.layer2 = nn.Sequential([
            Sign(in_channel),
            conv1x1(in_channel),
            nn.BatchNorm2d(in_channel)
        ])

    def forward(self, x):
        out = self.layer1(x)
        out = out + x
        out = self.layer2(x)
        return out + x

# Reduction Blocks for baseline model
class Base_Reduction_Block(nn.Module):
    def __init__(self, in_channel):
        self.layer1 = nn.Sequential(
            [
                Sign(in_channel),
                conv3x3(in_channel,in_channel, stride=2,padding=1),
                nn.BatchNorm2d(in_channel)
            ]
        )

        self.layer2_1 = nn.Sequential(
            [
                Sign(in_channel),
                conv1x1(in_channel,in_channel,stride=1),
                nn.BatchNorm2d(in_channel)
            ]
        )

        self.layer2_2 = nn.Sequential(
            [
                Sign(in_channel),
                conv1x1(in_channel,in_channel,stride=1),
                nn.BatchNorm2d(in_channel)
            ]
        )

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        out1 = self.layer1(x)
        pooled_x = self.pool(x)
        out1 = out1 + pooled_x

        out2_1 = self.layer2_1(out1)
        out2_1 = out2_1 + pooled_x

        out2_2 = self.layer2_2(out1)
        out2_2 = out2_2 + pooled_x

        return torch.cat([out2_1, out2_1], dim=1)




