import torch 
import numpy  as np
from torch import nn
from scipy import stats
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, r2_score
from torch.nn.modules import activation
from torch.nn.modules.dropout import Dropout
from _operator import linear_block, Self_Attention_for_GP
from Backbone import RL_regressor


class GP_net(RL_regressor):
    """
    A very special RL regressor model which spacial output of the last conv1d is collasped.
    """
    def __init__(self, 
                 channel_ls:list, 
                 kernel_size:list, 
                 stride:list, 
                 padding_ls:list,
                 diliation_ls:list,
                 pad_to:int = 100,
                 tower_width :int = 512,
                 dropout_rate : int = 0.3,
                 global_pooling:str='max',
                 activation:str='Mish'):
        super().__init__((channel_ls,kernel_size,stride,padding_ls,diliation_ls,pad_to), tower_width,dropout_rate, activation)

        # ------- Global pooling -------
        self.pool_fn = nn.MaxPool1d(self.out_length) if global_pooling=='max' else nn.AvgPool1d(self.out_length)
        
        # ------- linear block -------
        self.tower = linear_block(in_Chan=self.channel_ls[-1], out_Chan=512, dropout_rate=dropout_rate)
        self.fc_out = nn.Linear(tower_width,1)
    
    def forward_Global_pool(self, Z):
        # flatten
        batch_size = Z.shape[0]
        Z_flat = self.pool_fn(Z) 

        # pool and 
        if len(Z_flat.shape) == 3:
            Z_flat = Z.view(batch_size, self.channel_ls[-1])
        return Z_flat

    def forward_tower(self,Z):
        
        Z_flat = self.forward_Global_pool(Z)
        # tower part
        Z_to_out = self.tower(Z_flat)
        out = self.fc_out(Z_to_out)
        return out

class Atten_GP(GP_net):
    """
    repalce the final fc layers to self-atten, this allows for motif interaction 
    """
    def __init__(self, 
                 channel_ls:list, 
                 kernel_size:list, 
                 stride:list, 
                 padding_ls:list,
                 diliation_ls:list,
                 qk_dim:int = 64, 
                 n_head:int = 8,
                 pad_to:int = 100,
                 tower_width :int = 512,
                 dropout_rate : int = 0.3,
                 global_pooling:str='max',
                 activation:str='Mish'):

        super().__init__(channel_ls, kernel_size, stride, padding_ls, diliation_ls, pad_to, tower_width , dropout_rate , global_pooling, activation)

        self.attn_layer = Self_Attention_for_GP(channel_ls[-1], tower_width, qk_dim, n_head)
        self.after_norm = nn.SiLU()
        self.fc_out = nn.Linear(tower_width, 1)

    @torch.no_grad
    def _get_attention_map(self, X):
        """
        a function to quickly access attention matrix from input
        """
        Z = self.soft_share(X)
        Z_flat = self.forward_Global_pool(Z)
        attn, _ = self.attn_layer._get_attention_map(Z_flat) # this func has 2 return
        return attn

    def forward_tower(self,Z):
        # - this part is the same as GP -
        Z_flat = self.forward_Global_pool(Z)

        # -- Channel interaction attention --
        attn_out = self.attn_layer(Z_flat)
        attn_out = self.after_norm(attn_out)
        return self.fc_out(attn_out)