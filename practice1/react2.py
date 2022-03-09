from calendar import c
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def limit_conv_weight(member):
    if type(member) == GeneralConv2d:
        member.weight.data.clamp_(-1., 1.)

def limit_bn_weight(member):
    if type(member) == nn.BatchNorm2d:
        member.weight.data.abs_().clamp_(min=1e-2)

class Shift(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)
        self.out_channels = out_channels

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out       
    
    def __repr__(self):
        return f'{self.__class__.__name__}(out_channels={self.out_channels})'
    
class QuadraticSign(torch.autograd.Function):
    @staticmethod      
    def forward(ctx, input):
        grad_mask = ((input.lt(0) & input.ge(-1))*(2*input+2)+(input.lt(1) & input.ge(0))*(-2*input+2)).type(torch.float32)
        ctx.save_for_backward(grad_mask)                               
        return 2 * torch.ge(input, 0).type(torch.float32) - 1          

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors  
        return mask * grad_output
    

class ReAct_Sign(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.shift=Shift(in_channels=in_channels)
        
    def forward(self,x):
        y=self.shift(x)
        y=QuadraticSign.apply(y)
        return y

    
class ReAct_Relu(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.shift1=Shift(in_channels=in_channels)
        self.prelu=nn.PReLU(num_parameters=in_channels)
        self.shift2=Shift(in_channels=in_channels)
    
    def forward(self, x):
        y=self.shift1(x)
        y=self.prelu(y)
        y=self.shift2(y)

        return y

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.squeeze(x)
    

class DifferntiableSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        grad_mask=(input.lt(1)&input.ge(-1)).type(torch.float32)
        ctx.save_for_backward(grad_mask)
        
        return 2*torch.ge(input,0).type(torch.float32)-1

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return mask*grad_output
    
class GeneralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, conv, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding        
        self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True).to(device)
        self.conv = conv
        self.stride=stride

    def forward(self, x):
        real_weights = self.weight.view(self.shape)
        if self.conv == 'scaled_sign':
            scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True).to(device)
            y = F.conv2d(x, scaling_factor * DifferntiableSign.apply(real_weights), stride=self.stride, padding=self.padding).to(device)
        elif self.conv == 'sign':
            y = F.conv2d(x, DifferntiableSign.apply(real_weights), stride=self.stride, padding=self.padding).to(device)
        else:
            y = F.conv2d(x, real_weights, stride=self.stride, padding=self.padding).to(device)        
        return y

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride=1, padding={self.padding}, conv={self.conv})'    


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, alpha, beta1, beta2, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2 

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3= GeneralConv2d(inplanes, inplanes, stride=stride)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = GeneralConv2d(inplanes, planes)
        else:
            self.binary_pw_down1 = GeneralConv2d(inplanes, inplanes)
            self.binary_pw_down2 = GeneralConv2d(inplanes, inplanes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2,2)

    def forward(self, x):

        x_in = x*self.beta1 

        out1 = self.move11(x_in)
        out1 = self.binary_activation(out1)
        out1 = self.binary_3x3(out1)

        if self.stride == 2:
            x = self.pooling(x_in)

        out1 = x + out1*self.alpha

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out1_in = out1*self.beta2

        out2 = self.move21(out1_in)
        out2 = self.binary_activation(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = out2*self.alpha + out1

        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = out2_1*self.alpha + out1
            out2_2 = out2_2*self.alpha + out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2