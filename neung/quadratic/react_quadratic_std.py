import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from torch.nn.modules import loss

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
        self.block = Block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride,padding=padding,conv=conv)
        if(conv!='fc'):
            self.relu = nn.ReLU()
        self.conv=conv
         
    def forward(self, x):
        out=self.block(x)
        
        if self.conv!='fc':
            out=self.relu(out)          
        
        return out




class DWConvReact(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = conv
        self.depth = Block(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,stride=stride,padding=padding,conv=conv)
        self.point1 = Block(in_channels=in_channels, out_channels=in_channels,kernel_size=1, stride=1, padding=0, conv=conv)
        self.point2 = Block(in_channels=in_channels, out_channels=in_channels,kernel_size=1, stride=1, padding=0, conv=conv)
        self.rprelu11 = RPReLU(in_channels)
        self.rprelu21 = RPReLU(in_channels)
        self.rprelu22 = RPReLU(in_channels)
        self.pool = nn.AvgPool2d(kernel_size=2)

        if stride > 1:
            self.block = 'Reduction'
        else:
            self.block = 'Normal'

    def forward(self, x):
        # block -> shortcut -> RPReLU -> block -> shortcut -> RPReLU -> (concatenate)
        out = self.depth(x) # block
        
        if self.block == 'Reduction':
            x = self.pool(x)
        out = out + x # shortcut
        out1 = self.rprelu11(out) # RPReLU

        out_1 = self.point1(out1) # block
        out = out_1 + out    #shortcut
        out = self.rprelu21(out) # RPReLU
        
        if self.block == 'Reduction':
            out_2 = self.point2(out1) # block
            out_2 = out_2 + out    # shortcut
            out_2 = self.rprelu22(out_2) # RPReLU
            out = torch.cat([out, out_2],dim=1) # concatenate

        return out
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, conv={self.conv}, block={self.block})' 


class Block(nn.Sequential):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, conv):
        super().__init__()

        if conv == 'scaled_sign':
            self.add_layer(RSign(in_channels=in_channels))
        else:
            self.add_layer(Shift(in_channels=in_channels))
        ### conv + BN ###
        self.add_layer(GeneralConv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride, padding=padding, conv=conv))
        self.add_layer(nn.BatchNorm2d(out_channels))
        
    def add_layer(self, layer):
        self.add_module(layer.__class__.__name__, layer)


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        return QuadraticSign.apply(x)

class Distillation_loss(nn.Module):
    def __init__(self, balancing, temperature, classes):
        super(Distillation_loss,self).__init__()
        self.alpha=balancing
        self.T=temperature
        self.classes=classes
        
    def forward(self, y, logits, teacher_logits):      
        onehot_y=F.one_hot(y,num_classes=self.classes).type(torch.float32)
          
        loss1 = F.cross_entropy(onehot_y, F.softmax(logits,dim=1))
        loss2 = nn.KLDLoss(F.softmax(logits/self.T, dim=1),F.softmax(teacher_logits/self.T,dim=1))  
              
        return (1-self.alpha)*loss1+self.alpha*loss2

class ReactBase(LightningModule):
    def __init__(self, 
                 adam_init_lr=0.01, 
                 adam_weight_decay=0,
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
            loss_function=Distillation_loss(balancing=0.9, temperature=4, classes=10)
            loss=loss_function(y=y, logits=logits, teacher_logits=teacher_logits)
        
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