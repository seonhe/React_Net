import torch.nn as nn
import torch.nn.functional as F
import torch

from mobilenet import BCNNBase
from mobilenet import depthwise_separable_conv
from mobilenet import GeneralConv2d

class Model(BCNNBase):
    def __init__(self, structure, **kwargs):
        super().__init__(**kwargs)        
        self.blocks = nn.ModuleList([
            BasicBlock(in_channels=structure[i]['in_channels'],
                       out_channels=structure[i]['out_channels'],
                       kernel_size=structure[i]['kernel_size'],
                       stride=structure[i]['stride1'],
                       tride=structure[i]['stride2'],
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
                 stride1=1, stride2=1, 
                 padding=1, 
                 act_fn='relu',
                 dropout=0):
        super().__init__()
        
        if (in_channels > 3)&(in_channels<1024):
            self.add_layer(depthwise_separable_conv(nin=in_channels, nout=in_channels, kernel_size=3, kernels_per_layer=1, stride=stride1))                 
            self.add_layer(nn.BatchNorm2d(in_channels, affine=True))
            self.add_layer(nn.ReLU())

        elif (in_channels==1024)&(dropout==0):
            self.add_layer(depthwise_separable_conv(nin=in_channels, nout=in_channels, kernel_size=1, kernels_per_layer=1, stride=stride1))                 
            self.add_layer(nn.BatchNorm2d(in_channels, affine=True))
            self.add_layer(nn.ReLU())
        
        self.add_layer(GeneralConv2d(in_channels=in_channels, out_channels=out_channels, padding=padding, stride=stride2, kernel_size=kernel_size, conv=conv))                    
        self.add_layer(nn.BatchNorm2d(out_channels, affine=True))
        self.add_layer(nn.ReLU())
            
        if dropout > 0:
            self.add_layer(nn.Dropout(dropout))
    
    def add_layer(self, layer):
        self.add_module(layer.__class__.__name__, layer) #name(string), module