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
    def __init__(self, in_channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)
        self.out_channels = in_channels

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

class React_Base(LightningModule):
    def __init__(self, 
                 limit_conv_weight=True,
                 limit_bn_weight=True,
                 adam_init_lr=0.01, 
                 adam_weight_decay=0,
                 adam_betas=(0.9, 0.999),
                 lr_reduce_factor=0.1,
                 lr_patience=50,
                ):
        super().__init__()
        self.save_hyperparameters()
        # for logging the comp. graph
        #self.example_input_array = torch.ones((512, 3, 32, 32)) #########

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
 
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
    
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
        
    def on_train_batch_end(self, batch, batch_idx, unused):
        if self.hparams.limit_conv_weight:
            self.apply(limit_conv_weight)
        if self.hparams.limit_bn_weight:
            self.apply(limit_bn_weight)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.adam_init_lr,
            betas=self.hparams.adam_betas,
            weight_decay=self.hparams.adam_weight_decay,
        )
        scheduler_dict = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                factor=self.hparams.lr_reduce_factor, 
                patience=self.hparams.lr_patience,
            ),
            'interval': 'epoch',
            'monitor': 'val_acc',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}