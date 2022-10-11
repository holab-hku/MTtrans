import os, sys
import torch 
import numpy  as np
from torch import nn
from scipy import stats
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, r2_score
from torch.nn.modules import activation
from torch.nn.modules.dropout import Dropout
from ._operator import Conv1d_block, ConvTranspose1d_block, linear_block
    
class backbone_model(nn.Module):
    def __init__(self,conv_args,activation='ReLU'):
        """
        the most bottle model which define a soft-sharing convolution block some forward method 
        """
        super(backbone_model,self).__init__()
        channel_ls,kernel_size,stride,padding_ls,diliation_ls,pad_to = conv_args
        
        self.channel_ls = channel_ls
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_ls = padding_ls
        self.diliation_ls = diliation_ls
        self.pad_to = pad_to
        
        # model
        self.soft_share = Conv1d_block(channel_ls,kernel_size,stride,padding_ls,diliation_ls,activation=activation)
        # property
        self.stage = list(range(len(channel_ls)-1))
        self.out_length = self.soft_share.last_out_len(pad_to)
        self.out_dim = self.soft_share.last_out_len(pad_to)*channel_ls[-1]
    
    def _weight_initialize(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias) 
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
        elif isinstance(model, nn.Conv1d):
            nn.init.orthogonal_(model.weight)
        elif isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(model.weight, nonlinearity='leaky_relu',)
        elif isinstance(model, nn.BatchNorm1d):
            nn.init.constant_(model.weight, 1)
            nn.init.constant_(model.bias, 0)
    
    def forward_stage(self,X,stage):
        return self.soft_share.forward_stage(X,stage)
    
    def forward_tower(self,Z):
        """
        Each new backbone model should re-write the `forward_tower` method
        """
        return Z
    
    def forward(self,X):
        Z = self.soft_share(X)
        out = self.forward_tower(Z)
        return out
    
class RL_regressor(backbone_model):
    
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2, activation='ReLU'):
        """
        backbone for RL regressor task  ,the same soft share should be used among task
        Arguments:
            conv_args: (channel_ls,kernel_size,stride,padding_ls,diliation_ls)
        """
        super(RL_regressor,self).__init__(conv_args, activation)
        
        #  ------- architecture -------
        self.tower = linear_block(in_Chan=self.out_dim,out_Chan=tower_width,dropout_rate=dropout_rate)
        self.fc_out = nn.Linear(tower_width,1)
        
        #     ----- task specific -----
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.task_name = 'RL_regression'
        self.loss_dict_keys = ['Total']
        
    def forward_tower(self,Z):
        # flatten
        batch_size = Z.shape[0]
        Z_flat = Z.view(batch_size,-1)
        # tower part
        Z_to_out = self.tower(Z_flat)
        out = self.fc_out(Z_to_out)
        return out
    
    def squeeze_out_Y(self,out,Y):
        # ------ squeeze ------
        if len(Y.shape) == 2:
            Y = Y.squeeze(1)
        if len(out.shape) == 2:
            out = out.squeeze(1)
        
        assert Y.shape == out.shape, "keep label and pred the same shape"
        return out,Y
    
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
    
    def compute_loss(self,out,X,Y,popen):
        out,Y = self.squeeze_out_Y(out,Y)
        loss = self.loss_fn(out,Y) + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters()))) 
        return {"Total":loss}

class RL_gru(RL_regressor):
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2 ,activation='ReLU'):
        """
        tower is gru
        """      
        super().__init__(conv_args,tower_width,dropout_rate, activation)
        
        # previous, it is a linear layer
        if dropout_rate > 0 :
            self.soft_share.encoder = nn.ModuleList([
                nn.Sequential(conv_layer,nn.Dropout(dropout_rate)) 
                                for conv_layer in self.soft_share.encoder
             ])
        self.tower = nn.GRU(input_size=self.channel_ls[-1],
                            hidden_size=tower_width,
                            num_layers=2,
                            batch_first=True) # input : batch , seq , features
        self.fc_out = nn.Linear(tower_width,1)
        
        self.apply(self._weight_initialize)
        
    def forward_tower(self,Z):
        # flatten
        # batch_size = Z.shape[0]
        Z_flat = torch.transpose(Z,1,2)
        # tower part
        h_prim,(c1,c2) = self.tower(Z_flat)  # [B,L,h] , [B,h] cell of layer 1, [B,h] of layer 2
        out = self.fc_out(c2)
        return out

class RL_hard_share(RL_gru):
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2,activation='ReLU', tasks =['unmod1', 'human', 'vleng']):
        """
        Ribosome Loading Prediction with Hard-sharing;
        shared convolution bottom
        tower is gru
        """      
        super().__init__(conv_args,tower_width,dropout_rate,activation)
        self.all_tasks = tasks
        
        self.tower = nn.ModuleDict({task: self.tower_block(self.channel_ls[-1], tower_width) for task in self.all_tasks})
    
    def tower_block(self, c, w):
        block = nn.ModuleList([nn.GRU(input_size=c,
                                    hidden_size=w,
                                    num_layers=2,
                                    batch_first=True),
                                nn.Linear(w,1)])
        return block


    def forward(self, X):
        
        task = self.task # pass in cycle_train.py
        # Con block
        Z = self.soft_share(X)
        # tower
        Z_t = torch.transpose(Z, 1, 2)
        h_prim,(c1,c2) = self.tower[task][0](Z_t)
        out = self.tower[task][1](c2)
        
        return out
    
    def compute_loss(self,out,X,Y,popen):
        try:
            task_lambda = popen.chimera_weight
        except:
            task_lambda = {'unmod1':0.1, 'SubHuman':0.1, 'SubVleng':0.1,
                            'unmod1':0.1, 'human':0.1, 'vleng':0.1,  
                            'Andrev2015':1, 'muscle':1, 'pc3':1}
        
        loss_weight = task_lambda[self.task]
        out,Y = self.squeeze_out_Y(out,Y)
        loss = self.loss_fn(out,Y) + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters()))) 
        return {"Total":loss*loss_weight}

    def compute_acc(self,out,X,Y,popen=None):
        task = self.task
        Acc = super().compute_acc(out,X,Y,popen)['Acc']
        return {task+"_Acc" : Acc}

class RL_covar_reg(RL_hard_share):
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2, activation='ReLU',  n_covar=1, tasks =['unmod1', 'human', 'vleng']):
        """
        Ribosome Loading Prediction with Hard-sharing and account for covariates
        covariate is added at the last layer
        """      
        super().__init__(conv_args,tower_width,dropout_rate,activation, tasks)
        self.all_tasks = tasks
        self.n_covar = self.n_covar
        
        self.tower = nn.ModuleDict({task: tower_block(self.channel_ls[-1], tower_width) for task in self.all_tasks})

    def tower_block(self, c, w):
        block = nn.ModuleList([nn.GRU(input_size=c,
                                    hidden_size=w,
                                    num_layers=2,
                                    batch_first=True),

                                # covariate is added here
                                nn.Linear(w + self.n_covar,1)])
        return block
    
    def forward(self, X):
        
        task = self.task # pass in cycle_train.py
        X_seq, X_covar = X
        assert len(X_covar) == self.n_covar, "the # of covariates is not consistent with the model params"

        # Con block
        Z = self.soft_share(X_seq)
        # tower
        Z_t = torch.transpose(Z, 1, 2)
        h_prim,(c1,c2) = self.tower[task][0](Z_t)

        # concate
        linear_factor = torch.cat([c2, X_covar], dim=1)
        out = self.tower[task][1](linear_factor)
        
        return out

class RL_celline(RL_hard_share):
    def __init__(self,conv_args,tower_width,dropout_rate, tasks):
        super().__init__(conv_args,tower_width,dropout_rate, tasks)
    

class RL_3_data(RL_hard_share):
    def __init__(self,conv_args,tower_width,dropout_rate, tasks):
        super().__init__(conv_args,tower_width,dropout_rate, tasks)

class RL_6_data(RL_hard_share):
    def __init__(self,conv_args,tower_width,dropout_rate, tasks):
        super().__init__(conv_args,tower_width,dropout_rate, tasks)

    
class RL_FACS(RL_gru):
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2, n_bins=3):
        """
        tower is gru
        """      
        super().__init__(conv_args,tower_width,dropout_rate)
        
        self.fc_out = nn.Sequential(nn.Linear(tower_width,n_bins),
                                    nn.Softmax())
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self,out,X,Y,popen):
        
        loss = self.loss_fn(out,Y) + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters()))) 
        return {"Total":loss}    

class RL_mish_gru(RL_gru):
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2 ,activation='Mish'):
        """
        tower is gru
        """      
        super().__init__(conv_args,tower_width,dropout_rate, activation)

class RL_clf(RL_gru):

    def __init__(self,conv_args,n_calss,tower_width=40,dropout_rate=0.2):
        """
        transform RL gru into classifier
        """      
        super().__init__(conv_args,tower_width,dropout_rate)
        self.n_calss = n_calss
        # previous, it is a linear layer
        self.tower = nn.GRU(input_size=self.channel_ls[-1],
                            hidden_size=tower_width,
                            num_layers=2,
                            batch_first=True) # input : batch , seq , features
        self.fc_out = nn.Linear(tower_width,n_calss)
        
        self.apply(self._weight_initialize)
        
    def forward_tower(self,Z):
        # flatten
        # batch_size = Z.shape[0]
        Z_flat = torch.transpose(Z,1,2)
        # tower part
        h_prim,(c1,c2) = self.tower(Z_flat)  # [B,L,h] , [B,h] cell of layer 1, [B,h] of layer 2
        out = self.fc_out(c2)
        class_pred = torch.softmax(out,dim=1)
        return class_pred
    
    def compute_acc(self,out,X,Y,popen=None):
            
        # out,Y = self.squeeze_out_Y(out,Y)
        # error smaller than epsilon
        with torch.no_grad():
            acc = torch.sum(torch.argmax(out,dim=1) == Y.view(-1))/ Y.shape[0]
            # y_true = np.zeros((Y.shape[0],popen.n_class))
            
            # Y_int = np.array(Y.cpu().numpy(),dtype=np.int64)
            # for i,clas in enumerate(Y_int):
            #     y_true[i,int(clas)] = 1
                
            # auroc = roc_auc_score(Y_int,out.detach().cpu().numpy(),multi_class='ovr')
        return {"Acc":acc}
    
    def compute_loss(self,out,X,Y,popen):
        if len(Y.shape) >1:
            Y = Y.squeeze(dim=1).long()
        loss_fn=nn.CrossEntropyLoss()
        loss = loss_fn(out,Y) + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters()))) 
        return {"Total":loss}

class Reconstruction(backbone_model):
    def __init__(self,conv_args,variational=False,latent_dim=80):
        """
        the sequence reconstruction backbone
        """
        self.variational = variational
        self.latent_dim = latent_dim
        super(Reconstruction,self).__init__(conv_args)
        
        #  ------- architecture -------
        self.tower = ConvTranspose1d_block(*conv_args)
        
        #  ---- VAE only ----
        if self.variational == True:
            self.fc_mu = nn.Linear(self.out_dim,self.latent_dim)
            self.fc_sigma = nn.Linear(self.out_dim,self.latent_dim)
            self.fc_decode = nn.Linear(self.latent_dim,self.out_dim)
        
        #  ------- task specific -------
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.task_name = 'Reconstruction'
        self.loss_dict_keys = ['Total', 'MSE', 'KLD'] if self.variational else ['Total']
        
    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.Tensor) [B x D]
        """
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward_tower(self,Z):
        batch_size = Z.shape[0]
        re_code = Z
        mu = None
        sigma=None
        if self.variational:
             # project to N(µ, ∑)
            Z_flat = Z.view(batch_size,-1)
            mu = self.fc_mu(Z_flat)
            sigma = self.fc_sigma(Z_flat)
            code = self.reparameterize(mu,sigma)
            
            re_code = self.fc_decode(code)
            re_code = re_code.view(batch_size,self.soft_share.channel_ls[-1],self.out_length)    
        # decode
        recon_X = self.tower(re_code)
        
        return recon_X,mu,sigma
    
    # def forward(self,X):
        
    #     Z = self.soft_share(X)
    #     out = self.forward_tower(Z)
    #     recons_loss =self.loss_fn(out[0], X.transpose(1,2))
        
    #     with torch.no_grad():
    #         true_max=torch.argmax(X,dim=2)
    #         recon_max=torch.argmax(out[0],dim=1)
    #         acc =  torch.mean(torch.sum(true_max == recon_max,dim=1).float()).item()
    #     return out
    
    def compute_loss(self,out,X,Y,popen):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons,mu,sigma = out

        # Account for the minibatch samples from the dataset
        recons_loss =self.loss_fn(recons, X.transpose(1,2))
        loss =recons_loss + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters()))) 
        loss_dict = {'Total': loss}
        
        if self.variational:
            self.kld_weight = popen.kld_weight
            kld_loss = torch.mean(-0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim = 1), dim = 0)  # why is it negative ???
            loss = recons_loss + popen.kld_weight * kld_loss
            loss_dict = {'Total': loss, 'MSE':recons_loss, 'KLD':kld_loss}
        return loss_dict
    
    def compute_acc(self,out,X,Y,popen=None):
        """
        compute the reconstruction accuracy
        """
        pad_to = X.shape[1]
        recons,mu,sigma = out
        with torch.no_grad():
            true_max=torch.argmax(X,dim=2)
            recon_max=torch.argmax(recons,dim=1)
            acc =  torch.mean(torch.sum(true_max == recon_max,dim=1).float()).item()
        return {"Acc":acc / pad_to}

class Motif_detection(backbone_model):
    def __init__(self,conv_args,motifs:list,tower_width=40):
        """
        can detect different motif
        """
        super(Motif_detection,self).__init__(conv_args)
        self.num_labels = len(motifs)
        self.task_name = [seq + "_detection" for seq in motifs]
        
        # architecture
        self.tower_to_out = linear_block(in_Chan=self.out_dim,out_Chan=tower_width)
        self.fc_out = nn.Sequential(
            nn.Linear(tower_width,self.num_labels),
            nn.Sigmoid()
        )
        
        # task specific
        self.loss_fn = nn.BCELoss()
        self.loss_dict_keys = ['Total']
    
    def forward_tower(self,Z):
        """
        predicting several motifs at the same times
        """
        batch_size = Z.shape[0]
        Z_flat = Z.view(batch_size,-1)
        
        inter = self.tower_to_out(Z_flat)
        out = self.fc_out(inter)         # B * num_labels
        return out
    
    def compute_loss(self,out,X,Y,popen):
        loss = popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters())))
        
        # num of motifs
        for i in range(out.shape[1]):
            x = out[:,i]
            y = Y[:,i]
            loss += self.loss_fn(x,y)
        return {"Total":loss}
    
    def compute_acc(self,out,X,Y,popen=None):
        
        try:
            threshold = popen.threshold
        except:
            threshold = 0.5
            
        decision = out > threshold
        decision = decision.long()
        Y = Y.long()
        with torch.no_grad():
            acc = torch.sum(decision == Y).item() / (out.shape[0]*out.shape[1])
        return {"Acc":acc}
    
class Motif_detection_logit(Motif_detection):
    def __init__(self,conv_args,motifs:list,tower_width=40):
        """
        can detect different motif
        """
        super(Motif_detection,self).__init__(conv_args,motifs,tower_width)
        self.fc_out = nn.Linear(tower_width,self.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

class Multi_input_RL_regressor(RL_regressor):
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2,extra_input_num=1):
        super(Multi_input_RL_regressor,self).__init__(conv_args,tower_width,dropout_rate)

        self.merging_layer = nn.Sequential(
                            nn.Linear(tower_width+extra_input_num,tower_width),
                            nn.ReLU()
                            )
    
    def forward(self,X):
        if (type(X) == list) & (len(X) > 1):
            seq = X[0]
            other_input = [input.unsqueeze(1) for input in X[1:] if len(input.shape) == 1]

            Z = self.soft_share(seq)
            out = self.forward_tower(Z,other_input)
        else:
            Z = self.soft_share(X)
            out = self.forward_tower(Z)
        return out

    def forward_tower(self,Z,other_input):
        # flatten
        batch_size = Z.shape[0]
        Z_flat = Z.view(batch_size,-1)
        # tower part
        Z_to_out = self.tower(Z_flat)
        merge_info = torch.cat([Z_to_out]+other_input,dim=1)
        merge_to_out = self.merging_layer(merge_info)    # adding one layer to fully merge info from utr and categorical
        out = self.fc_out(merge_to_out)
        return out
