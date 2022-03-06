import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


from react import RSign
from react import Sign
from react import RPReLU
from react import GeneralConv2d
from react import ReactBase
from react import firstconv3x3


base_model = [{'in_channels':3, 'out_channels':32, 'stride':1 }, # block : react or baseline
            {'in_channels':32, 'out_channels':64, 'block':'baseline'}, # 16
            {'in_channels':64, 'out_channels':128, 'block':'baseline'}, # 8
            {'in_channels':128, 'out_channels':256, 'block':'baseline'}, # 4
            {'in_channels':256, 'out_channels':256, 'block':'baseline'}, # 4
            {'in_channels':256, 'out_channels':512, 'block':'baseline'}, # 1
            {'in_channels':512, 'out_channels':10, 'block':'baseline'}]


class TeacherModel(nn.Module):
    def __init__(self, structure):
        super().__init__()

        self.blocks = nn.ModuleList()
        conv1 = firstconv3x3(in_channels=structure[0]['in_channels'], 
                             out_channels=structure[0]['out_channels'],
                             stride=structure[0]['stride']
                             )
        self.blocks.append(conv1)

        for i in range(len(structure)-2):
            idx = i + 1
            if structure[idx]['in_channels'] == structure[idx]['out_channels']:
                self.blocks.append(
                    Normal_block(
                        structure[idx]['in_channels'],
                        structure[idx]['out_channels'],
                        structure[idx]['block']
                        )
                    )
            else:
                self.blocks.append(
                    Reduction_Block(
                        structure[idx]['in_channels'],
                        structure[idx]['out_channels'],
                        structure[idx]['block']
                    )
                )
        fc = GeneralConv2d(
            in_channels=structure[-1]['in_channels'],
            out_channels=structure[-1]['out_channels'],
            conv='scaled_sign',
            kernel_size=1,
            stride=1,
            padding=0
            )

        self.blocks.append(fc)
    
    def forward(self,x):
        for idx, block in enumerate(self.blocks):
            x = block(x)          
        return F.softmax(x.squeeze(dim=2).squeeze(dim=2), dim=1)





    

class ReactModel(ReactBase):
    def __init__(self, structure, **kwargs):
        super().__init__(**kwargs)

        self.blocks = nn.ModuleList()
        conv1 = firstconv3x3(in_channels=structure[0]['in_channels'], 
                             out_channels=structure[0]['out_channels'],
                             stride=structure[0]['stride']
                             )
        self.blocks.append(conv1)

        for i in range(len(structure)-2):
            idx = i + 1
            if structure[idx]['in_channels'] == structure[idx]['out_channels']:
                self.blocks.append(
                    Normal_block(
                        structure[idx]['in_channels'],
                        structure[idx]['out_channels'],
                        structure[idx]['block']
                        )
                    )
            else:
                self.blocks.append(
                    Reduction_Block(
                        structure[idx]['in_channels'],
                        structure[idx]['out_channels'],
                        structure[idx]['block']
                    )
                )
        fc = GeneralConv2d(
            in_channels=structure[-1]['in_channels'],
            out_channels=structure[-1]['out_channels'],
            conv='real',
            kernel_size=1,
            stride=1,
            padding=0
            )

        self.blocks.append(fc)

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)            
        return F.log_softmax(x.squeeze(dim=2).squeeze(dim=2), dim=1)

class Normal_block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 block
                 ):
        super().__init__()

        if block == 'baseline':
            act_fn = Sign(in_channels)
            self.conv = 'real'
        else:
            act_fn = RSign(in_channels)
            self.act_rprelu = RPReLU(in_channels)
            self.conv = 'scaled_sign'
        
        self.layer1 = nn.Sequential(
            act_fn,
            GeneralConv2d(in_channels=in_channels, out_channels=out_channels, conv=self.conv,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.layer2 = nn.Sequential(
            act_fn,
            GeneralConv2d(in_channels=out_channels, out_channels=out_channels, conv=self.conv,kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = out + x 
        

        if self.conv=='scaled_sign':
            out1 = self.act_rprelu(out)
        else:
            out1 = out

        out = self.layer2(out1)
        out2 = out + out1

        if self.conv=='scaled_sign':
            out3 = self.act_rprelu(out2)
        else:
            out3 = out2

        return out3
        

class Reduction_Block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 block
                 ):
        super().__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        

        if block == 'baseline':
            act_fn = Sign(in_channels)
            self.conv = 'real'
        else:
            act_fn = RSign(in_channels)
            self.in_rprelu = RPReLU(in_channels)
            
            self.conv = 'scaled_sign'

        self.layer1 = nn.Sequential(
            act_fn,
            GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv=self.conv,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(in_channels)
        )
        
        self.layer2_1 = nn.Sequential(
            act_fn,
            GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv=self.conv, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_channels)
        )

        self.layer2_2 = nn.Sequential(
            act_fn,
            GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv=self.conv, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_channels)
        )
        


    def forward(self, x):
        out = self.layer1(x)
        avg_out = self.avgpool(x)

        out1 = out + avg_out

        if self.conv=='scaled_sign':
            out2 = self.in_rprelu(out1)
        else:
            out2 = out1
        
        out2_1 = self.layer2_1(out2)
        out2_2 = self.layer2_2(out2)

        out3_1 = out2_1 + out2
        out3_2 = out2_2 + out2

        if self.conv=='scaled_sign':
            out4_1 = self.in_rprelu(out3_1)
            out4_2 = self.in_rprelu(out3_2)

        else:
            out4_1 = out3_1
            out4_2 = out3_2

        return torch.cat([out4_1, out4_2], dim=1)
        



