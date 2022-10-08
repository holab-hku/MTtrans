import torch 
import numpy  as np
from torch import nn
from scipy import stats
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, r2_score
from torch.nn.modules import activation
from torch.nn.modules.dropout import Dropout

class Conf_CNN(nn.Module):
    r"""
    representative learning for DNA motifs detection by Peter Koo et al
    https://doi.org/10.1371/journal.pcbi.1007560

    The model forces the motifs to be detected in the first CNN layer
    The certain receptive field set by different max pooling size
    """
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
            nns.append((f'MaxPool_{i}', nn.MaxPool1d(ps)))
            i += 1
        
        # 2 fc layers
        fcs = [
            ("fc_3", nn.Linear(channel_ls[-1], 512)),
            ("BN_3", nn.BatchNorm1d(512)),
            ("Act_3", nn.ReLU()),
            ("fc_out", nn.Linear(512, 1))

        ]

        self.network = nn.ModuleDict(
            {"Conv":nn.Sequential(OrderedDict(nns)), "fc":nn.Sequential(OrderedDict(fcs))}
            )
        self.loss_fn = nn.MSELoss()
            
    def __check_receptive_field(self):
        assert len(self.channel_ls) == 3
        k_len = len(self.kernel_size)
        st_len = len(self.stride)
        p_len = len(self.pool_size)
        

    def forward(self,X):
        """
        2 stage forward
        """
        Conv_out = self.network['Conv'](X)
        assert Conv_out.shape[-1] == 1, "the maxpooling is not restricting values to 1"

        out = self.network['fc'](Conv_out.squeeze(dim=-1))
        return out
    
    def compute_loss(self, out,X,Y,popen):
        
        if len(Y.shape) == 2:
            Y = Y.squeeze(1)
        if len(out.shape) == 2:
            out = out.squeeze(1)

        return self.loss_fn(out, Y)
    
    def compute_acc(self,out,X,Y,popen=None):
        try:
            epsilon = popen.epsilon
        except:
            epsilon = 0.3
            
        out,Y = self.squeeze_out_Y(out,Y)
        # error smaller than epsilon
        with torch.no_grad():
            y_ay = Y.cpu().numpy()
            out_ay = out.cpu().numpy()
            # acc = torch.sum(torch.abs(Y-out) < epsilon).item() / Y.shape[0]
            acc = stats.spearmanr(y_ay,out_ay)[0]
            # acc = r2_score(y_ay, out_ay)
        return {"Acc":acc}