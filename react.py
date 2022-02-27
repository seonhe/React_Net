import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DifferentiableSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        grad_mask = (input.lt(1) & input.ge(-1)).type(torch.float32)        
        ctx.save_for_backward(grad_mask)
        return 2 * torch.gt(input, 0).type(torch.float32) - 1

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return mask * grad_output    

class Sign(nn.Module):
    def __init__(self, out_channel):
        self.out_channel = out_channel

    def forward(self, x):
        out = DifferentiableSign(self.out_channel).apply(x)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(out_channels={self.out_channel})'


# RSign : shift + sign
class RSign(nn.Module):
    def __init__(self, out_channel):
        self.out_channel = out_channel
        
    def forward(self, x):
        out = Shift(self.out_channel).apply(x)
        out = Sign(self.out_channel).apply(out)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(out_channels={self.out_channel})'

# RPReLU : Shift + PReLU + Shift
class RPReLU(nn.Module):
    def __init__(self, out_channel):
        self.out_channel = out_channel
        self.prelu = nn.PReLU(self.out_channel, init=0.25)

    def forward(self,x):
        out = Shift(self.out_channel).apply(x) # x - gamma
        out = self.prelu(out)
        out = Shift(self.out_channel).apply(x) # PReLU - zeta
        return out
    
    def __repr__(self):
        return f'{self.__class__.__name__}(out_channels={self.out_channel})'


class GeneralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, conv, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding        
        self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.conv = conv

    def forward(self, x):
        real_weights = self.weight.view(self.shape)
        if self.conv == 'scaled_sign':
            scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            y = F.conv2d(x, scaling_factor * Sign.apply(real_weights), stride=self.stride, padding=self.padding)
        elif self.conv == 'sign':
            y = F.conv2d(x, Sign.apply(real_weights), stride=self.stride, padding=self.padding)
        else:
            y = F.conv2d(x, real_weights, stride=self.stride, padding=self.padding)        
        return y

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, conv={self.conv})'    
    



