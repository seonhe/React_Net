import torch.nn as nn
import torch.nn.functional as F
import torch

from bcnn import BCNNBase
from bcnn import GeneralConv2d
from bcnn import BinaryActivation
from bcnn import Shift
from bcnn import Clamp

class Model(BCNNBase):
    def __init__(self, structure, **kwargs):
        super().__init__(**kwargs)        
        self.blocks = nn.ModuleList([
            BasicBlock(in_channels=structure[i]['in_channels'],
                       out_channels=structure[i]['out_channels'],
                       kernel_size=structure[i]['kernel_size'],
                       stride=structure[i]['stride'],
                       padding=structure[i]['padding'],
                       act_fn=structure[i]['act_fn'],
                       conv=structure[i]['conv'],
                       dropout=structure[i]['dropout'])
            for i in range(len(structure))
        ])

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)            
        return F.log_softmax(x.squeeze(dim=2).squeeze(dim=2), dim=1)


class BasicBlock(nn.Sequential):
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
        
#         self.conv = BinaryConv2d(in_channels=in_channels, out_channels=out_channels, padding=padding, kernel_size=kernel_size)
#         if stride > 1: self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0)
#         self.bn = nn.BatchNorm2d(out_channels, affine=True)
#         if act_fn == 'sign':
#             self.shift = Shift(out_channels=out_channels)
#             self.sign = BinaryActivation()
#         if dropout > 0: self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         y = self.conv(x)
#         if hasattr(self, 'pool'): y = self.pool(y)
#         y = self.bn(y)
#         if hasattr(self, 'shift'): y = self.shift(y)
#         if hasattr(self, 'sign'): y = self.sign(y)
#         # if hasattr(self, 'sign'): y = self.sign(torch.clamp(y, min=-1, max=1))
#         if hasattr(self, 'dropout'): y = self.dropout(y)
#         return y

# class BasicBlock(nn.Sequential):
#     def __init__(self, 
#                  in_channels, 
#                  out_channels, 
#                  kernel_size, 
#                  stride=1, 
#                  padding=1, 
#                  act_fn='sign', 
#                  dropout=0):
#         super().__init__()
#         self.add_module('conv', BinaryConv2d(in_channels=in_channels, out_channels=out_channels, padding=padding, kernel_size=kernel_size))
#         if stride > 1: self.add_module('pool', nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0))
#         self.add_module('bn', nn.BatchNorm2d(out_channels, momentum=0.1, affine=True))
#         if act_fn == 'sign': self.add_module('sign', BinaryActivation())
#         if dropout > 0: self.add_module('dropout', nn.Dropout(dropout))