import torch.nn as nn
import torch.nn.functional as F
import torch

from bcnn_gap import BCNNBase
from bcnn_gap import GeneralConv2d
from bcnn_gap import BinaryActivation
from bcnn_gap import Shift
from bcnn_gap import Clamp

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
                       dropout=structure[i]['dropout'],
                       avgpool=structure[i]['avgpool'])
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
                 dropout=0,
                 avgpool='no'):
        super().__init__()
        
        self.add_layer(GeneralConv2d(in_channels=in_channels, out_channels=out_channels, padding=padding, kernel_size=kernel_size, conv=conv))                    
        if stride > 1: 
            self.add_layer(nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0))
        
        self.add_layer(nn.BatchNorm2d(out_channels,affine=False))
        

        
        if act_fn == 'sign':
            self.add_layer(Shift(out_channels=out_channels))
            self.add_layer(Clamp())
            self.add_layer(BinaryActivation())
        elif act_fn == 'relu':
            self.add_layer(nn.ReLU())


        if avgpool=='pool':
            self.add_layer(nn.AvgPool2d(kernel_size=4))
            self.add_layer(BinaryActivation())


        if dropout > 0:
            self.add_layer(nn.Dropout(dropout))
    
    def add_layer(self, layer):
        self.add_module(layer.__class__.__name__, layer)
        