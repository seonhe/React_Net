
import torch.nn as nn
import torch.nn.functional as F
import torch

from react2 import React_Base
from react2 import ReAct_Sign
from react2 import ReAct_Relu
from react2 import GeneralConv2d
from react2 import Squeeze


class Model(React_Base):
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
            print(x.shape)            
        return F.log_softmax(x.squeeze(dim=2).squeeze(dim=2), dim=1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act_fn, conv, dropout):
        super(BasicBlock, self).__init__()
       
        if ((in_channels == 3)|(in_channels == 1024)):
            self.sign1 = ReAct_Sign(in_channels=in_channels)
            self.binary_3x3= GeneralConv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding, conv=conv)
            self.batchnorm1=nn.BatchNorm2d(num_features=out_channels)
            self.relu1=ReAct_Relu(in_channels=out_channels)
        else:
            self.sign1 = ReAct_Sign(in_channels=in_channels)
            self.binary_3x3= GeneralConv2d(in_channels=in_channels, out_channels=in_channels, stride=stride, kernel_size=kernel_size, padding=padding, conv=conv)
            self.batchnorm1=nn.BatchNorm2d(num_features=in_channels)
            self.relu1=ReAct_Relu(in_channels=in_channels)
            

        if in_channels == out_channels:
            self.sign2 = ReAct_Sign(in_channels=in_channels)
            self.binary_1x1= GeneralConv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=1, padding=0, conv='scaled sign')
            self.batchnorm2=nn.BatchNorm2d(num_features=out_channels)
            self.relu2=ReAct_Relu(in_channels=in_channels)
        else:
            self.sign2_1 = ReAct_Sign(in_channels=in_channels)
            self.sign2_2 = ReAct_Sign(in_channels=in_channels)
            
            self.binary_1x1_1= GeneralConv2d(in_channels=in_channels, out_channels=in_channels, stride=1, kernel_size=1, padding=0, conv='scaled sign')
            self.binary_1x1_2= GeneralConv2d(in_channels=in_channels, out_channels=in_channels, stride=1, kernel_size=1, padding=0, conv='scaled sign')
            
            self.batchnorm2_1=nn.BatchNorm2d(num_features=in_channels)
            self.batchnorm2_2=nn.BatchNorm2d(num_features=in_channels)
            
            self.relu2_1=ReAct_Relu(in_channels=in_channels)
            self.relu2_2=ReAct_Relu(in_channels=in_channels)

            self.pooling = nn.AvgPool2d(2,2)
            
        self.in_channels=in_channels
        self.out_channels=out_channels  
        self.dropout=nn.Dropout(dropout)        
        

    def forward(self, x):
        out0=x
        
        if (self.in_channels==1024)|(self.out_channels==1024):
            out1=self.sign1(out0)
            out1=self.binary_3x3(out1)
            out3=self.batchnorm1(out1)
            
        elif (self.in_channels==3)|(self.in_channels==1024):
            out1=self.binary_3x3(out0)
            out3=self.batchnorm1(out1)
            
        else:        
            out1=self.sign1(out0)
            out1=self.binary_3x3(out1)
            out1=self.batchnorm1(out1)
        
            if self.in_channels!=self.out_channels:
                out0=self.pooling(out0)
            
            out1=out0+out1
            out1=self.relu1(out1)
        
            if self.in_channels==self.out_channels:
                out2=self.sign2(out1)
                out2=self.binary_1x1(out2)
                out2=self.batchnorm2(out2)
                out3=out1+out2
                out3=self.relu2(out3)
            else:
                out2_1=self.sign2_1(out1)
                out2_1=self.binary_1x1_1(out2_1)
                out2_1=self.batchnorm2_1(out2_1)
                out2_1=self.relu2_1(out2_1)
            
                out2_2=self.sign2_2(out1)
                out2_2=self.binary_1x1_2(out2_2)
                out2_2=self.batchnorm2_2(out2_2)
                out2_2=self.relu2_2(out2_2)
            
                out3 = torch.cat([out2_1, out2_2], dim=1)
                
        out3=self.dropout(out3)
        return out3