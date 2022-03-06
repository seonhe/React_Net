import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


from react import RSign
from react import GeneralConv2d
from react import ReactBase
from react import firstconv3x3
from react import DWConvReact
from react import DWConvReal
torch.use_deterministic_algorithms(True)



# in_channels, out_channels, kernel_size, stride, padding, conv

# baseline {'in_channels':128, 'out_channels':128, 'stride':2, 'kernel_size':3, 'padding':1, 'conv':'scaled_sign', 'act_fn':'sign', 'dropout':0},

class ReactModel(ReactBase):
    def __init__(self, structure, **kwargs):
        super().__init__(structure, kwargs)        
        self.blocks = nn.ModuleList()

        for i in range(len(structure)):
            if i == 0:
                self.blocks.append(firstconv3x3( # conv + bn
                in_channels=structure[i]['in_channels'],
                out_channels=structure[i]['out_channels'],
                stride=structure[i]['stride'],
                conv=structure[i]['conv']
                )
            )
            else:
                if structure[i]['conv'] == 'real': # Real block
                    self.blocks.append(
                        DWConvReal(
                            in_channels=structure[i]['in_channels'],
                            out_channels =structure[i]['out_channels'],
                            kernel_size=structure[i]['kernel_size'],
                            stride=structure[i]['stride'],
                            padding=structure[i]['padding'],
                            conv=structure[i]['conv']
                        )
                    )

                elif structure[i]['conv'] == 'scaled_sign': # scaled.block
                    self.blocks.append(
                        DWConvReact(
                            in_channels=structure[i]['in_channels'],
                            out_channels =structure[i]['out_channels'],
                            kernel_size=structure[i]['kernel_size'],
                            stride=structure[i]['stride'],
                            padding=structure[i]['padding'],
                            conv=structure[i]['conv']
                        )
                    )

                elif structure[i]['conv'] == 'pool':
                    self.blocks.append(nn.AvgPool2d(
                        kernel_size=structure[i]['kernel_size'],
                        stride=structure[i]['stride']
                        )
                    )

        


    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)            
        return F.log_softmax(x.squeeze(dim=2).squeeze(dim=2), dim=1)


        