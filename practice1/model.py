from msvcrt import kbhit
import torch.nn as nn
import torch.nn.functional as F
import torch

from react import BCNNBase
from react import React_Sign
from react import ProduceParameter
from react import React_PReLu
from react import GeneralConv2d


class Model(BCNNBase):
    def __init__(self, structure, **kwargs):
        super().__init__(**kwargs)        
        self.blocks = nn.ModuleList([GeneralConv2d(in_channels=3, out_channels=128,
                        conv='sign', kernel_size=3, padding=1, stride=1)])
        self.structure=structure

    def forward(self, x):
        for i in range(len(self.structure)):
            if(self.structure[i]['in_channels']==self.structure[i]['out_channels']):
                self.blocks.append(Normal_Block(in_channels=self.structure[i]['in_channels'],
                    out_channels=self.structure[i]['out_channels'],
                    kernel_size1=self.structure[i]['kernel_size1'],
                    kernel_size2=self.structure[i]['kernel_size2'],
                    stride1=self.structure[i]['stride1'],
                    stride2=self.structure[i]['stride2'],
                    padding1=self.structure[i]['padding1'],
                    padding2=self.structure[i]['padding2'],
                    conv=self.structure[i]['conv'],
                    dropout=self.structure[i]['dropout']))
            else:
                self.blocks.append(Reduction_Block(in_channels=self.structure[i]['in_channels'],
                    out_channels=self.structure[i]['out_channels'],
                    kernel_size1=self.structure[i]['kernel_size1'],
                    kernel_size2=self.structure[i]['kernel_size2'],
                    stride1=self.structure[i]['stride1'],
                    stride2=self.structure[i]['stride2'],
                    padding1=self.structure[i]['padding1'],
                    padding2=self.structure[i]['padding2'],
                    conv=self.structure[i]['conv'],
                    dropout=self.structure[i]['dropout']))
            
        for idx, block in enumerate(self.blocks):
            x = block(x)            
        return F.log_softmax(x.squeeze(dim=2).squeeze(dim=2), dim=1)

'''
class Normal_Block(nn.Sequential):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 conv,
                 stride=1, 
                 padding=1, 
                 act_fn='sign',
                 dropout=0):
        super().__init__()
        
        self.add_layer(GeneralConv2d(in_channels=in_channels, out_channels=out_channels, padding=padding, kernel_size=kernel_size, conv=conv))                    
        if stride > 1: 
            self.add_layer(nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0))
            
        self.add_layer(nn.BatchNorm2d(out_channels, affine=True))
        
        if act_fn == 'sign':
            self.add_layer(Shift(out_channels=out_channels))
            self.add_layer(Clamp())
            self.add_layer(BinaryActivation())
        elif act_fn == 'relu':
            self.add_layer(nn.ReLU())
            
        if dropout > 0:
            self.add_layer(nn.Dropout(dropout))
    
    def add_layer(self, layer):
        self.add_module(layer.__class__.__name__, layer) #name(string), module
'''       

class Normal_Block(nn.Sequential):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size1, kernel_size2,
                 conv,
                 stride1=1, stride2=1,
                 padding1=1, padding2=1,
                 dropout=0):
        super().__init__()
              
        self.Rsign1=React_Sign(in_channels=in_channels)
        self.Conv1=GeneralConv2d(in_channels=in_channels, out_channels=out_channels,
                        conv=conv, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        self.x_bias1, self.y_bias1, self.inclination1 =ProduceParameter(in_channels=in_channels)
        
        self.Rsign2=React_Sign(in_channels=in_channels)
        self.Conv2=GeneralConv2d(in_channels=in_channels, out_channels=out_channels,
                        conv=conv, kernel_size=kernel_size2, padding=padding2, stride=stride2)
        self.x_bias2, self.y_bias2, self.inclination2 =ProduceParameter(in_channels=in_channels)
        
        self.bn=nn.BatchNorm2d(in_channels=in_channels, affine=True)
    
    def forward(self, x):
        y=self.Rsign1(x)
        y=self.Conv1(y)
        y=self.bn(y)
        y=y+x
        
        y=React_PReLu(y, self.x_bias1, self.y_bias1, self.inclination1)
        k=y
        
        k=self.Rsign2(k)
        k=self.Conv1(k)
        k=self.bn(k)
        
        k=k+y
        
        return React_PReLu(k, self.x_bias2, self.y_bias2, self.inclination2)
    
class Reduction_Block(nn.Sequential):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size1, kernel_size2,
                 conv,
                 stride1=1, stride2=1,
                 padding1=1, padding2=1,
                 dropout=0):
        super().__init__()
              
        self.Rsign1=React_Sign(in_channels=in_channels)
        self.Conv1=GeneralConv2d(in_channels=in_channels, out_channels=out_channels,
                        conv=conv, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        
        self.avgpool=nn.AvgPool2d(kernel_size=2, stride=2, padding=0)########????padding
        
        self.x_bias1, self.y_bias1, self.inclination1 =ProduceParameter(in_channels=in_channels)
        
        self.Rsign2=React_Sign(in_channels=in_channels)
        self.Conv2=GeneralConv2d(in_channels=in_channels, out_channels=out_channels,
                        conv=conv, kernel_size=kernel_size2, padding=padding2, stride=stride2)
        self.x_bias2, self.y_bias2, self.inclination2 =ProduceParameter(in_channels=in_channels)
        
        self.Rsign3=React_Sign(in_channels=in_channels)
        self.Conv3=GeneralConv2d(in_channels=in_channels, out_channels=out_channels,
                        conv=conv, kernel_size=kernel_size2, padding=padding2, stride=stride2)
        self.x_bias3, self.y_bias3, self.inclination3 =ProduceParameter(in_channels=in_channels)
        
        self.bn=nn.BatchNorm2d(in_channels=in_channels, affine=True)
    
    def forward(self, x):
        y=self.Rsign1(x)
        y=self.Conv1(y)
        y=self.bn(y)
        
        y=y+self.avgpool(x)
        y=React_PReLu(y, self.x_bias1, self.y_bias1, self.inclination1)
        
        k1=self.Rsign2(y)
        k1=self.Conv1(k1)
        k1=self.bn(k1)
        k1=k1+y        
        k1=React_PReLu(k1, self.x_bias2, self.y_bias2, self.inclination2)
        
        k2=self.Rsign2(y)
        k2=self.Conv1(k2)
        k2=self.bn(k2)
        k2=k2+y
        k2=React_PReLu(k2, self.x_bias3, self.y_bias3, self.inclination3)
        
        return torch.cat((k1,k2),dim=1)
        
        
        
        
        
        
        
