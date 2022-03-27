import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from torch.nn.modules import loss

torch.use_deterministic_algorithms(True)

def limit_conv_weight(member):
    if type(member) == GeneralConv2d:
        member.weight.data.clamp_(-1., 1.)

def limit_bn_weight(member):
    if type(member) == nn.BatchNorm2d:
        member.weight.data.abs_().clamp_(min=1e-2)
    
class Clamp(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-1, max=1)
    
    def __repr__(self):
        return f'{self.__class__.__name__}'


class Shift(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)
        self.in_channels = in_channels

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out       
    
    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels})'

    
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

# RSign : shift + sign
class RSign(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.shift = Shift(self.in_channels)
        
    def forward(self, x):
        out = self.shift(x)
        out = QuadraticSign.apply(out)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels})'

# RPReLU : Shift + PReLU + Shift
class RPReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = Shift(self.in_channels)
        self.prelu = nn.PReLU(self.in_channels, init=0.25)
        self.zeta = Shift(self.in_channels)

    def forward(self,x):
        out = self.gamma(x) # x - gamma
        out = self.prelu(out)
        out = self.zeta(out) # PReLU - zeta
        return out
    
    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels})'



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
            y = F.conv2d(x, scaling_factor * QuadraticSign.apply(real_weights), stride=self.stride, padding=self.padding)
        elif self.conv == 'sign':
            y = F.conv2d(x, QuadraticSign.apply(real_weights), stride=self.stride, padding=self.padding)
        else: # real
            y = F.conv2d(x, real_weights, stride=self.stride, padding=self.padding)        
        return y

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, conv={self.conv})'    


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv):
        super().__init__()
        self.conv=GeneralConv2d(in_channels=in_channels, out_channels=out_channels, conv=conv, kernel_size=kernel_size, stride=stride, padding=padding)
        
        if(out_channels!=10):
            self.bn=nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        else:
            self.avgpool=nn.AvgPool2d(kernel_size=2, stride=1)

        self.out_channels=out_channels
         
    def forward(self, x):
        if self.out_channels==10:
          out=self.avgpool(x)
          out=self.conv(out)
        else:
          out=self.conv(x)
          out=self.bn(out)    
          out=self.relu(out)
   
        return out 

        
class Block(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, conv, reduction=None):
        super().__init__()
        
        self.rsign=RSign(in_channels=in_channels)
        self.conv=GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv=conv, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn=nn.BatchNorm2d(in_channels)
        self.rprelu=RPReLU(in_channels=in_channels)
        self.reduction=reduction
        
        if(reduction!=None):
            self.avgpool=nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.avgpool=nn.AvgPool2d(kernel_size=1, stride=1)
            
    def forward(self, x):
        out=self.rsign(x)
        out=self.conv(out)
        out=self.bn(out)
        out=out+self.avgpool(x) 
        out=self.rprelu(out)
        
        return out
    
class Block1(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, conv, reduction=None):
        super().__init__()
        
        self.rsign=RSign(in_channels=in_channels)
        self.conv=GeneralConv2d(in_channels=in_channels, out_channels=in_channels, conv=conv, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn=nn.BatchNorm2d(in_channels)
        self.rprelu=RPReLU(in_channels=in_channels)
        self.reduction=reduction
        
        if(reduction!=None):
            self.avgpool=nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.avgpool=nn.AvgPool2d(kernel_size=1, stride=1)
            
    def forward(self, x):
        out=self.rsign(x)
        out=self.conv(out)
        out=self.bn(out)
        out=out+self.avgpool(x) 
        out=self.rprelu(out)
        
        return out
    
class Concatenate(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, conv, reduction=None):
        super().__init__()
        
        self.block1=Block(in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, conv=conv, reduction=reduction)
        self.block2=Block(in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, conv=conv, reduction=reduction)
    
    def forward(self, x):
        block1=self.block1(x)
        block2=self.block2(x)
        out=torch.cat([block1, block2],dim=1)
        return out
    
class Normal_Block(nn.Sequential):
    def __init__(self,in_channels, kernel_size, stride, padding, conv):
        super().__init__()

        self.add_layer(Block(in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, conv=conv, reduction=None))
        
        self.add_layer(Block1(in_channels=in_channels, kernel_size=1, stride=1, padding=0, conv=conv, reduction=None))
        
    def add_layer(self, layer):
        self.add_module(layer.__class__.__name__, layer)
        

class Reduction_Block(nn.Sequential):
    def __init__(self,in_channels, kernel_size, stride, padding, conv):
        super().__init__()

        self.add_layer(Block(in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, conv=conv, reduction='Yes'))
        
        self.add_layer(Concatenate(in_channels=in_channels, kernel_size=1, stride=1, padding=0, conv=conv, reduction=None))
        
    def add_layer(self, layer):
        self.add_module(layer.__class__.__name__, layer)
    
        
        
class Distillation_loss(nn.Module):
    def __init__(self, classes, batch_size):
        super(Distillation_loss,self).__init__()
        self.c=classes
        self.n=batch_size
        
    def forward(self, logits, teacher_logits):      

        loss = torch.sum(torch.sum(teacher_logits*torch.log(logits/teacher_logits),dim=0),dim=0).type(torch.float32)
        loss=-loss/self.n
              
        return loss

class ReactBase(LightningModule):
    def __init__(self, 
                 adam_init_lr=0.01, 
                 adam_weight_decay=0.00001,
                 adam_betas=(0.9, 0.999),
                 lr_reduce_factor=0.1,
                 lr_patience=50,
                 limit_conv_weight=True,
                 limit_bn_weight=True,
                 teacher_model=None
                ):
        super().__init__()
        self.save_hyperparameters()
        # for logging the comp. graph
        #self.example_input_array = torch.ones((512, 3, 32, 32))
        self.teacher_model=teacher_model

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        if self.teacher_model==None:
                loss = F.nll_loss(logits, y)
        
        else:
            teacher_logits=self.teacher_model(x).detach()
            loss_function=Distillation_loss(classes=10, batch_size=512)
            loss=loss_function(logits=logits, teacher_logits=teacher_logits)
        
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