
import torch.nn as nn
import torch.nn.functional as F
import torch

from react1 import BCNNBase
from react1 import ReAct_Sign
from react1 import ReAct_Relu
from react1 import GeneralConv2d
from react1 import Squeeze

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model(BCNNBase):
    def __init__(self, structure, **kwargs):
        super().__init__(**kwargs)        
        self.blocks = nn.ModuleList([])
        self.structure=structure

    def forward(self, x):
        self.blocks.append(GeneralConv2d(in_channels=3, out_channels=32,
                        conv='scaled_sign', kernel_size=3, padding=1, stride=1))
        
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
                
        self.blocks.append(Squeeze())
        self.blocks.append(nn.Linear(in_features=1024, out_features=10))
        
        print("start")
        for idx, block in enumerate(self.blocks):
            x = block(x)
            print(idx)
            print(x.shape)  
        print(x)
        return F.log_softmax(x, dim=1)
      

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
        self.Rsign1=ReAct_Sign(in_channels=in_channels)
        self.Conv1=GeneralConv2d(in_channels=in_channels, out_channels=out_channels, conv=conv, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        self.RPRelu1=ReAct_Relu(in_channels=in_channels)
        
        self.Rsign2=ReAct_Sign(in_channels=in_channels)
        self.Conv2=GeneralConv2d(in_channels=in_channels, out_channels=out_channels, conv=conv, kernel_size=kernel_size2, padding=padding2, stride=stride2)
        self.RPRelu2=ReAct_Relu(in_channels=out_channels)
        
        self.bn1=nn.BatchNorm2d(num_features=in_channels, affine=True)
        self.bn2=nn.BatchNorm2d(num_features=out_channels, affine=True)
    
    def forward(self, x):
        y=self.Rsign1(x)
        y=self.Conv1(y)
        y=self.bn1(y)
        y=y+x
        
        y=self.RPRelu1(y)
        
        k=self.Rsign2(y)
        k=self.Conv2(k)
        k=self.bn2(k)
        
        k=k+y
        
        return self.RPRelu2(k)
    
class Reduction_Block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size1, kernel_size2,
                 conv,
                 stride1=1, stride2=1,
                 padding1=1, padding2=1,
                 dropout=0):
        super().__init__()
              
        self.Rsign1=ReAct_Sign(in_channels=in_channels)
        self.Conv1=GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv=conv, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        self.avgpool=nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.RPRelu1=ReAct_Relu(in_channels=in_channels)
        
        self.Rsign2=ReAct_Sign(in_channels=in_channels)
        self.Conv2=GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv=conv, kernel_size=kernel_size2, padding=padding2, stride=stride2)
        self.RPRelu2=ReAct_Relu(in_channels=in_channels)
        
        self.Rsign3=ReAct_Sign(in_channels=in_channels)
        self.Conv3=GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv=conv, kernel_size=kernel_size2, padding=padding2, stride=stride2)
        self.RPRelu3=ReAct_Relu(in_channels=in_channels)
        
        self.bn1=nn.BatchNorm2d(num_features=in_channels, affine=True)
        self.bn2=nn.BatchNorm2d(num_features=in_channels, affine=True)
        self.bn3=nn.BatchNorm2d(num_features=in_channels, affine=True)
    
    def forward(self, x):
        y=self.Rsign1(x)
        y=self.Conv1(y)
        y=self.bn1(y)
        
        y=y+self.avgpool(x)
        y=self.RPRelu1(y)
        
        k1=self.Rsign2(y)
        k1=self.Conv2(k1)
        k1=self.bn2(k1)
        k1=k1+y        
        k1=self.RPRelu2(k1)
        
        k2=self.Rsign3(y)
        k2=self.Conv3(k2)
        k2=self.bn3(k2)
        k2=k2+y
        k2=self.RPRelu3(k2)
        
        return torch.cat((k1,k2),dim=1)
    
        
        
        
        
        
        
        
