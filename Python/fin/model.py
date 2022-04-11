import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule



from react import Conv
from react import ReactBase
from react import Normal_Block
from react import Reduction_Block
from react import Concatenate

torch.use_deterministic_algorithms(True)



class ReactModel(ReactBase):
    def __init__(self, structure, **kwargs):
        super().__init__(structure, kwargs)        
        self.blocks = nn.ModuleList()

        for i in range(len(structure)):
            if i == 0:
                self.blocks.append(
                    Conv( 
                            in_channels=structure[i]['in_channels'],
                            out_channels =structure[i]['out_channels'],
                            kernel_size=structure[i]['kernel_size'],
                            stride=structure[i]['stride'],
                            padding=structure[i]['padding'],
                            conv=structure[i]['conv']
                    )
                )
            
            elif structure[i]['conv'] == 'fc':
                self.blocks.append(
                        Conv(
                            in_channels=structure[i]['in_channels'],
                            out_channels =structure[i]['out_channels'],
                            kernel_size=structure[i]['kernel_size'],
                            stride=structure[i]['stride'],
                            padding=structure[i]['padding'],
                            conv='real',
                    )
                )
                
            else:
                if structure[i]['in_channels']==structure[i]['out_channels']:
                    self.blocks.append(
                        Normal_Block(
                            in_channels=structure[i]['in_channels'],
                            kernel_size=structure[i]['kernel_size'],
                            stride=structure[i]['stride'],
                            padding=structure[i]['padding'],
                            conv=structure[i]['conv'],

                    )
                )
                    
                else:
                    self.blocks.append(
                        Reduction_Block(
                            in_channels=structure[i]['in_channels'],
                            kernel_size=structure[i]['kernel_size'],
                            stride=structure[i]['stride'],
                            padding=structure[i]['padding'],
                            conv=structure[i]['conv'],
                        )
                    )        
                    
        self.blocks.append(nn.Dropout(structure[i]['dropout']))


    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return F.log_softmax(x.squeeze(dim=2).squeeze(dim=2),dim=1)
 
class T_ReactModel(nn.Module):
    def __init__(self, structure, **kwargs):
        super().__init__(structure, kwargs)        
        self.blocks = nn.ModuleList()

        for i in range(len(structure)):
            if i == 0:
                self.blocks.append(
                    Conv( 
                            in_channels=structure[i]['in_channels'],
                            out_channels =structure[i]['out_channels'],
                            kernel_size=structure[i]['kernel_size1'],
                            stride=structure[i]['stride1'],
                            padding=structure[i]['padding'],
                            conv=structure[i]['conv']
                    )
                )
            
            elif structure[i]['conv'] == 'fc':
                self.blocks.append(
                        Conv(
                            in_channels=structure[i]['in_channels'],
                            out_channels =structure[i]['out_channels'],
                            kernel_size=structure[i]['kernel_size1'],
                            stride=structure[i]['stride1'],
                            padding=structure[i]['padding'],
                            conv='real',
                    )
                )
                
            else:
                if structure[i]['in_channels']==structure[i]['out_channels']:
                    self.blocks.append(
                        Normal_Block(
                            in_channels=structure[i]['in_channels'],
                            kernel_size=structure[i]['kernel_size'],
                            stride=structure[i]['stride'],
                            padding=structure[i]['padding'],
                            conv=structure[i]['conv'],
                            reduction=None
                    )
                )
                    
                else:
                    self.blocks.append(
                        Reduction_Block(
                            in_channels=structure[i]['in_channels'],
                            kernel_size=structure[i]['kernel_size'],
                            stride=structure[i]['stride'],
                            padding=structure[i]['padding'],
                            conv=structure[i]['conv'],
                            reduction='Yes'
                        )
                    )        
                    
        self.blocks.append(nn.Dropout(structure[i]['dropout']))


    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return F.log_softmax(x.squeeze(dim=2).squeeze(dim=2),dim=1)