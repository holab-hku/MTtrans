import os
import sys
import torch
from torch import nn
import numpy as np
from .CNN_models import Conv_AE,Conv_VAE,cal_conv_shape
from .Self_attention import self_attention

class Baseline(nn.Module):
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,dropout_ls,num_label,loss_fn,pad_to):
        """
        Conv - Dense Framework DL Regressor to predict ribosome load (rl) from 5' UTR sequence
        Arguments:
        ...channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label : parameter to define the Conv layers
        ...loss_fn : regression loss function `MSELoss` or `MyHingeLoss`
        """
        super(Baseline,self).__init__()
        #         ==<<|  properties  |>>==
        self.kernel_size = kernel_size
        self.loss_fn = loss_fn
        self.channel_ls = channel_ls
        self.padding_ls = padding_ls
        self.diliat_ls =  diliat_ls
        self.L_in = pad_to
        self.loss_dict_keys = ['Total','MAE','R.M.S.E']
        # the basic element block of CNN
        
        #         ==<<|  Conv layers  |>>==
        # 3 layers in the 5'UTR paper
        self.encoder = nn.ModuleList(
            [self.Conv_block(channel_ls[i],channel_ls[i+1],padding_ls[i],diliat_ls[i],dropout_rate=dropout_ls[i]) for i in range(len(channel_ls)-1)]
        )        
        
        # compute the output shape of conv layer
        self.out_len = int(self.compute_out_dim(kernel_size,L_in=self.L_in))
        self.out_dim = self.out_len * channel_ls[-1]
        
        #         ==<<|  Dense layers  |>>==
        self.fc_out = nn.Sequential( 
            nn.Linear(self.out_dim,40),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(40,num_label)
        )
        
        self.define_loss()  
        self.acc_hinge = MyHingeLoss(None)
        
        self.apply(self.weight_initialize)
        self.rl_posi = None       # single output
        
    def weight_initialize(self, model):
        if type(model) in [nn.Linear]:
        	nn.init.xavier_uniform_(model.weight)
        	nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
        elif isinstance(model, nn.Conv1d):
            nn.init.kaiming_normal_(model.weight, nonlinearity='leaky_relu',)
        elif isinstance(model, nn.BatchNorm1d):
            nn.init.constant_(model.weight, 1)
            nn.init.constant_(model.bias, 0)
    
    def Conv_block(self,inChan,outChan,padding,diliation,stride=1,dropout_rate=0): 
        """
        Building Block of stack conv
        """
        net = nn.Sequential(
                    nn.Conv1d(inChan,outChan,self.kernel_size,stride=1,padding=padding,dilation=diliation),
                    # nn.BatchNorm1d(outChan),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate))
        return net
    
    def define_loss(self):
        if self.loss_fn in ['mse','MSE']:
            self.regression_loss = nn.MSELoss(reduction='mean')
        else:
            self.regression_loss = MyHingeLoss(reduction='mean')
        
    
    def encode(self,X):
        
        X = X.transpose(1,2)  # to B*4*100
        Z = X
        for model in self.encoder:
            Z = model(Z)
        return Z
    
    def forward(self,X):
        """
        Conv -> flatten -> Dense
        """
        batch_size = X.shape[0]
        
        z = self.encode(X)
        z_flat = z.view(batch_size,-1)
        out = self.fc_out(z_flat)       # no activation for the last layer
        
        return out
        
        
    def compute_acc(self,out,X,Y,popen):
        """
        for this regression task, accuracy  is the percentage that prediction error < epsilon (Lambda) 
        """
        
        epsilon = popen.epsilon
        
        batch_size = Y.shape[0]
        rl_pred = out[self.rl_posi] if type(out) == tuple else out
        if rl_pred.shape != Y.shape:
            if len(rl_pred.shape) == 2:
                rl_pred = rl_pred.squeeze()
            if len(Y.shape) == 2:
                Y = Y.squeeze()
        with torch.no_grad(): 
            loss = self.acc_hinge(rl_pred,Y,epsilon=0.3)
            n_inrange = (loss==0).squeeze().sum().item()
        
        return {"Acc":n_inrange / batch_size}
       
        
    def compute_loss(self,out,X,Y,popen):
        """
        it's termed `chimela_loss` to keep compatability with MTL MODELS (the same dataset was used)
        `chiemela_loss` requires three input : 
        ...out:
        ...Y:
        ...Lambda : which is the epsilon of hinge loss if possible
        """
        self.Lambda = popen.chimerla_weight
        batch_size = Y.shape[0]
        rl_pred = out[self.rl_posi] if type(out) == tuple else out
        
        #  Hinge or MSE
        if self.loss_fn in ['mse','MSE']:
            loss = self.regression_loss(rl_pred,Y)
        else:
            loss = self.regression_loss(rl_pred,Y,Lambda).squeeze()
        with torch.no_grad():
            MAE = torch.abs(rl_pred-Y).mean()                                  # Mean Absolute Error
            RMSE = torch.sqrt( torch.sum((rl_pred-Y)**2) / batch_size)         # Root Mean Square Error
        return {"Total":loss,"MAE":MAE,"R.M.S.E":RMSE}
    
    def compute_out_dim(self,kernel_size,L_in = 100,num_layer=None):
        """
        manually compute the final length of convolved sequence
        """
        loop_times = num_layer if type(num_layer) == int else len(self.channel_ls)-1
        for i in range(loop_times):
            L_out = cal_conv_shape(L_in,kernel_size,stride=1,padding=self.padding_ls[i],diliation=self.diliat_ls[i])
            L_in = L_out
        return L_out


class Hi_baseline(Baseline):
    
    def __init__(self,channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label,loss_fn):
        """
        Hierarchical Multi-task baseline model, first detect is there uAUG in the sequence and then predict rl score
        ...chimerla_weight: list len = 3, [w1,w2,epsilon]
        """
        super(Hi_baseline,self).__init__(channel_ls,padding_ls,diliat_ls,latent_dim,kernel_size,num_label,loss_fn)
        
        self.first_outdim = int(self.compute_out_dim(kernel_size,num_layer=1))  # 1 but not 0 , i.e 4 -> 120
        
        self.uAUG_dk = 64                                        # dimension of key
        self.uAUG_task_wk = nn.Linear(64,self.first_outdim)      # the task query for uAUG
        self.uAUG_to_key = nn.Linear(channel_ls[1],64)           # 
        
        self.uAUG_fc_out = nn.Sequential(
            nn.Linear(self.first_outdim*channel_ls[1],128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128,1),
            nn.Sigmoid()                                        # Binary classifier
        ) 
        
        self.rl_predictor = nn.Sequential( 
            nn.Linear(self.out_dim,39),
            nn.Dropout(0.4),
            nn.ReLU()
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(40,128),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128,num_label)
        )
        
        self.uAUG_loss = nn.BCELoss(reduction='mean')
        self.rl_posi = 1
        self.loss_dict_keys = ['Total','MAE','R.M.S.E','BCE']
        
    def forward(self,X):
        
        batch_size = X.shape[0]
        if X.shape[1] == 100:
            X = X.transpose(1,2)  # to B*4*100
        
        
        z1 = self.encoder[0](X)                           #  i.e. B * channal * self.first_out
        z1_T = z1.transpose(1,2)                          #  i.e. B * self.first_out *  channal
        
        # --- uAUG task ---
        # generate task specific attention matrix
        uAUG_key = self.uAUG_to_key(z1_T)                 #  i.e.  B * self.first_out * 64
        uAUG_ATT_M = self.uAUG_task_wk(uAUG_key)          # attention matrix : B * self.first_out * self.first_out
        
        # reweight with attention and predict
        uAUG_reatt_f = torch.bmm(z1,uAUG_ATT_M).view(batch_size,-1)      
        uAUG = self.uAUG_fc_out(uAUG_reatt_f)        
        
        # --- regression task ---
        # encode
        Z = z1
        for layer in self.encoder[1:]:
            Z = layer(Z)
        Z_flat = Z.view(batch_size,-1)
        
        # 1 part of predictor
        # intermediate 
        rl_inter = self.rl_predictor(Z_flat)
        
        rl_and_uAUG = torch.cat([rl_inter,uAUG],dim=1)
        rl = self.fc_out(rl_and_uAUG)
        
        out = uAUG,rl
        
        return out

    def compute_loss(self,out,X,Y,popen):
        """
        it's termed `chimela_loss` to keep compatability with MTL MODELS (the same dataset was used)
        `chiemela_loss` requires three input : 
        ...out:
        ...Y:
        ...Lambda : which is the epsilon of hinge loss if possible
        """
        self.Lambda = popen.chimerla_weight
        batch_size = Y.shape[0]
        uAUG_pred, rl_pred = out
        rl_true,uAUG_true = Y[:,0],Y[:,1]
        # uAUG_true = AUG_true.long()
        
        #  Hinge or MSE
        if self.loss_fn in ['mse','MSE']:
            loss = self.regression_loss(rl_pred,Y)
        else:
            epsilon = Lambda[2]
            loss = self.regression_loss(rl_pred,Y,epsilon).squeeze()
        uAUG_loss =self.uAUG_loss(uAUG_pred,uAUG_true)
        with torch.no_grad():
            MAE = torch.abs(rl_pred-Y).mean()                                  # Mean Absolute Error
            RMSE = torch.sqrt( torch.sum((rl_pred-Y)**2) / batch_size)         # Root Mean Square Error
        
        total_loss = Lambda[0]*uAUG_loss + Lambda[1]*loss
        return {"Total":total_loss,"Reg_loss":loss,"MAE":MAE,"R.M.S.E":RMSE,'BCE':uAUG_loss}
    
    def compute_acc(self,out,X,Y,popen):
        """
        compute the accuracy of TE range class prediction
        """
        uAUG_pred, TE_pred = out
        TE_true,uAUG_true = Y[:,0],Y[:,1]
        batch_size = TE_true.shape[0]
        epsilon = 0.2
        with torch.no_grad():
            pred = torch.sum(torch.abs(TE_pred - TE_true) < epsilon).item()
            # else:
            #     pred = torch.sum(torch.argmax(TE_pred,dim=1) == TE_true).item()
            
        return {"Acc":pred / batch_size}

class Hiep_baseline(Hi_baseline):
    def __init__(self):
        """
        try MMoe with hierarchical sharing
        """
    

class MyHingeLoss(torch.nn.Module):

    def __init__(self,reduction='mean'):
        self.reduction = reduction
        super(MyHingeLoss, self).__init__()

    def forward(self, out, Y,epsilon):
        """
        linear version hinge regression loss 
        """
        hinge_loss = torch.abs(out-Y) - epsilon
        hinge_loss[hinge_loss < 0] = 0
        if self.reduction  == 'mean':
            hinge_loss = torch.mean(hinge_loss)
        return hinge_loss