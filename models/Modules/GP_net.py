import torch 
import numpy  as np
from torch import nn
from scipy import stats
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, r2_score
from torch.nn.modules import activation
from torch.nn.modules.dropout import Dropout


class GP_net(nn.Module):
    def __init__(self, channel_ls, kernel_size=[8,5], stride=[1,1], pool_size=[10,5]):
        super().__init__()

        self.channel_ls =channel_ls 
        CNN_dims = list(zip(channel_ls[:-1], channel_ls[1:]))
        self.CNN_dims = CNN_dims

        self.kernel_size = kernel_size 
        self.stride = stride
        self.pool_size = pool_size

        # 2 CNN layer
        nns = []
        i = 1
        for in_out , ks, strid, ps in zip(CNN_dims, kernel_size, stride, pool_size):
            layer = nn.Sequential(
                nn.Conv1d(*in_out, ks, strid),
                nn.BatchNorm1d(in_out[1]),
                nn.Mish()
            )
            nns.append((f'Conv_{i}', layer))
            i += 1