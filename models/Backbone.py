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
        
        assert Y.shape == out.shape
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
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2, tasks =['unmod1', 'human', 'vleng']):
        """
        tower is gru
        """      
        super().__init__(conv_args,tower_width,dropout_rate)
        self.all_tasks = tasks
        tower_block = lambda c,w : nn.ModuleList([nn.GRU(input_size=c,
                                                        hidden_size=w,
                                                        num_layers=2,
                                                        batch_first=True),
                                                nn.Linear(w,1)])
        
        self.tower = nn.ModuleDict({task: tower_block(self.channel_ls[-1], tower_width) for task in self.all_tasks})
    
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
