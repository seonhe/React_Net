from numpy import block
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


from react import RPReLU, RSign
from react import GeneralConv2d
from react import ReactBase
from react import firstconv3x3






class Model(ReactBase):
    def __init__(self, structure, **kwargs):
        super().__init__(structure, kwargs)
        self.blocks = nn.ModuleList()
        for i in range(len(structure)):
            if i == 0:
                self.blocks.append(
                    firstconv3x3(
                        in_channels=structure[i]['in_channels'],
                        out_channels=structure[i]['out_channels'],
                        stride=structure[i]['stride'], 
                        padding=structure[i]['padding'],
                        kernel_size=structure[i]['kernel_size']
                        
                    )
                )

            elif(i != 0 and structure[i]['stride'] == 1) : # Normal block
                self.blocks.append(
                    BasicBlock(
                        in_channels=structure[i]['in_channels'],
                        kernel_size=structure[i]['kernel_size'],
                        stride=structure[i]['stride'],
                        padding=structure[i]['padding']
                    )
                )
                self.blocks.append(
                    BasicBlock(
                        in_channels=structure[i]['in_channels'],
                        kernel_size=1,
                        stride=1,
                        padding=0
                    )
                )
        
            elif(i != 0 and structure[i]['stride'] == 2) : # Normal block
                self.blocks.append(
                    BasicBlock(
                        in_channels=structure[i]['in_channels'],
                        kernel_size=structure[i]['kernel_size'],
                        stride=structure[i]['stride'],
                        padding=structure[i]['padding']
                    )
                )
                self.blocks.append(
                    Duplicate(
                        in_channels=structure[i]['in_channels']
                    )
                )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,10)
        
    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x,dim=1)


class BasicBlock(nn.Module):
    def __init__(self,in_channels,kernel_size,stride,padding):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.rsign = RSign(in_channels=in_channels)
        self.conv = GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv="scaled_sign",kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(in_channels)
        self.rprelu = RPReLU(in_channels=in_channels)

    def forward(self, x):
        pooled_out = self.pool(x)
        rsign_out = self.rsign(x)
        conv_out = self.conv(rsign_out)
        bn_out = self.bn(conv_out)
        shortcut = pooled_out + bn_out
        out = self.rprelu(shortcut)

        return out

class Duplicate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.basicblock1 = BasicBlock(in_channels, kernel_size=1, stride=1, padding=0)
        self.basicblock2 = BasicBlock(in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.basicblock1(x)
        out2 = self.basicblock2(x)

        out = torch.cat([out1, out2], dim=1)

        return out
        


            


