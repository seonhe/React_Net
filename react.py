import torch
import torch.nn as nn

# RSign : shift + sign

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


class Sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        grad_mask = (input.lt(1) & input.ge(-1)).type(torch.float32)        
        ctx.save_for_backward(grad_mask)
        return 2 * torch.gt(input, 0).type(torch.float32) - 1

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return mask * grad_output    


class RSign(nn.Module):
    def __init__(self, out_channel):
        self.out_channel = out_channel
        
    def forward(self, x):
        out = Shift(self.out_channel).apply(x)
        out = Sign(self.out_channel).apply(out)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(out_channels={self.out_channels})'


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
        return f'{self.__class__.__name__}(out_channels={self.out_channels})'




