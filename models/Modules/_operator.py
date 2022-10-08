
import torch 
import numpy  as np
from torch import nn
from scipy import stats
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, r2_score
from torch.nn.modules import activation
from torch.nn.modules.dropout import Dropout

class Conv1d_block(nn.Module):
    """
    the Convolution backbone define by a list of convolution block
    """
    def __init__(self,channel_ls,kernel_size,stride, padding_ls=None,diliation_ls=None,pad_to=None, activation='ReLU'):
        """
        Argument
            channel_ls : list, [int] , channel for each conv layer
            kernel_size : int
            stride :  list , [int]
            padding_ls :   list , [int]
            diliation_ls : list , [int]
        """
        super(Conv1d_block,self).__init__()
        ### property
        self.activation = activation
        self.channel_ls = channel_ls
        self.kernel_size = kernel_size
        self.stride = stride
        if padding_ls is None:
            self.padding_ls = [0] * (len(channel_ls) - 1)
        else:
            assert len(padding_ls) == len(channel_ls) - 1
            self.padding_ls = padding_ls
        if diliation_ls is None:
            self.diliation_ls = [1] * (len(channel_ls) - 1)
        else:
            assert len(diliation_ls) == len(channel_ls) - 1
            self.diliation_ls = diliation_ls
        
        self.encoder = nn.ModuleList(
            #                   in_C         out_C           padding            diliation
            [self.Conv_block(channel_ls[i],channel_ls[i+1],self.padding_ls[i],self.diliation_ls[i],self.stride[i]) for i in range(len(self.padding_ls))]
        )
        
    def Conv_block(self,in_Chan,out_Chan,padding,dilation,stride): 
        
        activation_layer = eval(f"nn.{self.activation}")
        
        block = nn.Sequential(
                nn.Conv1d(in_Chan,out_Chan,self.kernel_size,stride,padding,dilation),
                nn.BatchNorm1d(out_Chan),
                activation_layer())
        
        return block
    
    def forward(self,x):
        if x.shape[2] == 4:
            out = x.transpose(1,2)
        else:
            out = x
        for block in self.encoder:
            out = block(out)
        return out
    
    def forward_stage(self,x,stage):
        """
        return the activation of each stage for exchanging information
        """
        assert stage < len(self.encoder)
        
        out = self.encoder[stage](x)
        return out

    def cal_out_shape(self,L_in=100,padding=0,diliation=1,stride=2):
        """
        For convolution 1D encoding , compute the final length 
        """
        L_out = 1+ (L_in + 2*padding -diliation*(self.kernel_size-1) -1)/stride
        return L_out
    
    def last_out_len(self,L_in=100):
        for i in range(len(self.padding_ls)):
            padding = self.padding_ls[i]
            diliation = self.diliation_ls[i]
            stride = self.stride[i]
            L_in = self.cal_out_shape(L_in,padding,diliation,stride)
        # assert int(L_in) == L_in , "convolution out shape is not int"
        
        return int(L_in) if L_in >=0  else 1
    
class ConvTranspose1d_block(Conv1d_block):
    """
    the Convolution transpose backbone define by a list of convolution block
    """
    def __init__(self,channel_ls,kernel_size,stride,padding_ls=None,diliation_ls=None,pad_to=None):
        channel_ls = channel_ls[::-1]
        stride = stride[::-1]
        padding_ls =  padding_ls[::-1] if padding_ls  is not None else  [0] * (len(channel_ls) - 1)
        diliation_ls =  diliation_ls[::-1] if diliation_ls  is not None else  [1] * (len(channel_ls) - 1)
        super(ConvTranspose1d_block,self).__init__(channel_ls,kernel_size,stride,padding_ls,diliation_ls,pad_to)
        
    def Conv_block(self,in_Chan,out_Chan,padding,dilation,stride): 
        """
        replace `Conv1d` with `ConvTranspose1d`
        """
        block = nn.Sequential(
                nn.ConvTranspose1d(in_Chan,out_Chan,self.kernel_size,stride,padding,dilation=dilation),
                nn.BatchNorm1d(out_Chan),
                nn.ReLU())
        
        return block
    
    def cal_out_shape(self,L_in,padding=0,diliation=1,stride=1,out_padding=0):
        #                  L_in=100,padding=0,diliation=1,stride=2
        """
        For convolution Transpose 1D decoding , compute the final length
        """
        L_out = (L_in -1 )*stride + diliation*(self.kernel_size -1 )+1-2*padding + out_padding 
        return L_out


class linear_block(nn.Module):
    def __init__(self,in_Chan,out_Chan,dropout_rate=0.2):
        """
        building block func to define dose network
        """
        super(linear_block,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_Chan,out_Chan),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.BatchNorm1d(out_Chan)
        )
    def forward(self,x):
        return self.block(x)