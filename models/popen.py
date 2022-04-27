import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import utils
import json
from models import DL_models
from models import CNN_models
from models import MTL_models
from models import Baseline_models
from models import Backbone
from models import Cross_stitch
import configparser
import logging

class Auto_popen(object):
    def __init__(self,config_file):
        """
        read the config_fiel
        """
        # machine config path
        self.shuffle = True
        self.script_dir = utils.script_dir
        self.data_dir = utils.data_dir
        self.log_dir = utils.log_dir
        self.pth_dir = utils.pth_dir
        self.set_attr_as_none(['te_net_l2','loss_fn','modual_to_fix','other_input_columns','pretrain_pth'])
        self.split_like_paper = False
        self.loss_schema = 'constant'
        
        # transform to dict and convert to  specific data type
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.config_file = config_file
        self.config_dict = {item[0]: eval(item[1]) for item in self.config.items('DEFAULT')}
        
        # assign some attr from config_dict         
        self.set_attr_from_dict(self.config_dict.keys())
        self.check_run_and_setting_name()                          # check run name
        self._dataset = "_" + self.dataset if self.dataset != '' else self.dataset
        # the saving direction
        self.path_category = self.config_file.split('/')[-4]
        self.vae_log_path = config_file.replace('.ini','.log')
        self.vae_pth_path = os.path.join(self.pth_dir,self.model_type+self._dataset,self.setting_name,self.run_name + '-model_best.pth')
        self.Resumable = False
        
        # generate self.model_args
        self.get_model_config()
    
    def set_attr_from_dict(self,attr_ls):
        for attr in attr_ls:
            self.__setattr__(attr,self.config_dict[attr])
    
    def set_attr_as_none(self,attr_ls):
        for attr in attr_ls:
            self.__setattr__(attr,None)

    def check_run_and_setting_name(self):
        file_name = self.config_file.split("/")[-1]
        dir_name = self.config_file.split("/")[-2]
        self.setting_name = dir_name
        assert self.run_name == file_name.split(".")[0]
        
    def get_model_config(self):
        """
        assert we type in the correct model type and group them into model_args
        """
        if "Conv" in self.model_type:
            assert self.model_type in dir(CNN_models), "model type not in CNN models"
            self.Model_Class = eval("CNN_models.{}".format(self.model_type))
        elif "LSTM" in self.model_type:
            assert self.model_type in dir(DL_models), "model type not in DL models"
            self.Model_Class = eval("DL_models.{}".format(self.model_type))
        else:
            if self.model_type in dir(Backbone):
                self.Model_Class = eval("Backbone.{}".format(self.model_type))
            elif self.model_type in dir(Baseline_models):
                self.Model_Class = eval("Baseline_models.{}".format(self.model_type))
            elif self.model_type in dir(Cross_stitch):
                self.Model_Class = eval("Cross_stitch.{}".format(self.model_type))
            else:
                assert self.model_type in dir(MTL_models), "model type not in MTL models"
                self.Model_Class = eval("MTL_models.{}".format(self.model_type))
        
        # teacher foring
        # if self.teacher_forcing is True:
        #     # default setting
        #     self.teacher_forcing = True
        #     t_k,t_b = (0.032188758248682,0.032188758248682)  # k = b , k = log5 / 50
        # elif type(self.config_dict['teacher_forcing']) == list:
        #     self.teacher_forcing = True
        #     t_k,t_b = self.config_dict['teacher_forcing']
        # elif self.config_dict['teacher_forcing'] == 'fixed':
        #     t_k = t_b = 100
        # elif self.config_dict['teacher_forcing'] == False:
        #     t_k = t_b = 100
            
        
        # if "LSTM" in self.model_type:
        #     self.model_args=[self.input_size,
        #                      self.config_dict["hidden_size_enc"],
        #                      self.config_dict["hidden_size_dec"],
        #                      self.config_dict["num_layers"],
        #                      self.config_dict["latent_dim"],
        #                      self.config_dict["seq_in_dim"],
        #                      self.config_dict["decode_type"],
        #                      self.config_dict['teacher_forcing'],
        #                      self.config_dict['discretize_input'],
        #                      t_k,t_b,
        #                      self.config_dict["bidirectional"],
        #                      self.config_dict["fc_output"]]
        
        # model_args = {k: v for k, v in args.items() if
        #           k in [p.name for p in inspect.signature(Model.__init__).parameters.values()]}
            
        if "Conv" in self.model_type:
            args_to_read = ["channel_ls","padding_ls","diliat_ls","latent_dim","kernel_size"]
            self.model_args=[self.__getattribute__(args) for args in args_to_read]
        
        if  self.model_type in  ['TO_SEQ_TE','TRANSFORMER_SEQ_TE','TRANSFORMER_SEQ_RL']:
            args_to_read = ["channel_ls","padding_ls","diliat_ls","latent_dim","kernel_size","num_label"]
            self.model_args=[self.__getattribute__(args) for args in args_to_read] 
            
        if self.model_type  in  ['Baseline','Hi_baseline']:
            args_to_read = ["channel_ls","padding_ls","diliat_ls","latent_dim","kernel_size","dropout_ls","num_label","loss_fn","pad_to"]
            self.model_args=[self.__getattribute__(args) for args in args_to_read] 
        
        if "TWO_TASK_AT" in self.model_type:
            args_to_read = ["latent_dim","linear_chann_ls","num_label","te_chann_ls","ss_chann_ls","dropout_rate"]
            self.model_args=[self.__getattribute__(args) for args in args_to_read]

        if self.model_type in ['RL_clf','RL_gru', 'RL_FACS', 'RL_mish_gru', 'RL_regressor','Reconstruction','Motif_detection','RL_hard_share','RL_3_data', 'RL_celline', 'RL_6_data']:           # Backbone
            # conv_args define the soft-sharing part
            conv_args = ["channel_ls","kernel_size","stride","padding_ls","diliation_ls","pad_to"]
            self.conv_args = tuple([self.__getattribute__(arg) for arg in conv_args])
            
            # left args dfine the tower part in which the arguments are different among tasks
            left_args={'RL_regressor':["tower_width","dropout_rate"],
                       'RL_clf':["n_class","tower_width","dropout_rate"],
                       'RL_gru':["tower_width","dropout_rate"],
                       'RL_FACS': ["tower_width","dropout_rate"],
                       'RL_hard_share':["tower_width","dropout_rate","cycle_set" ],
                       'RL_3_data':["tower_width","dropout_rate","cycle_set" ],
                       'RL_celline':["tower_width","dropout_rate","cycle_set"],
                       'RL_6_data':["tower_width","dropout_rate","cycle_set"],
                       'RL_mish_gru':["tower_width","dropout_rate"],
                       'Reconstruction':["variational","latent_dim"],
                       'Motif_detection':["aux_task_columns","tower_width"]}[self.model_type]
            
            self.model_args = [self.conv_args] + [self.__getattribute__(arg) for arg in left_args]
        
        if self.model_type in ['Multi_input_RL_regressor']:
            conv_args = ["channel_ls","kernel_size","stride","padding_ls","diliation_ls","pad_to"]
            self.conv_args = tuple([self.__getattribute__(arg) for arg in conv_args])

            # extra_input_col = len(other_input_columns)
            self.model_args = [self.conv_args]+[self.tower_width,self.dropout_rate,len(self.other_input_columns)]
        
        if self.model_type == 'CrossStitch':
            
            self.model_args = [self.__getattribute__(arg) for arg in ['tasks','alpha','beta']]
        
        if self.model_type in ['Cross_stitch_classifier', 'MLP_down','MLP_linear_reg']:
            
            self.model_args = [self.channel_ls, self.tower_width]
        
            
    def check_experiment(self,logger):
        """
        check any unfinished experiment ?
        """
        log_save_dir = os.path.dirname(self.vae_log_path)
        pth_save_dir = os.path.join(self.pth_dir,self.model_type+self._dataset,self.setting_name)
        # make dirs 
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        if not os.path.exists(pth_save_dir):
            os.makedirs(pth_save_dir)
        
        # check resume
        if os.path.exists(self.vae_log_path) & os.path.exists(self.vae_pth_path):
            self.Resumable = True
            logger.info(' \t \t ==============<<<  Experiment detected  >>>============== \t \t \n')
            
    def update_ini_file(self,E,logger):
        """
        E  is the dict contain the things to update
        """
        # update the ini file
        self.config_dict.update(E)
        strconfig = {K: repr(V) for K,V in self.config_dict.items()}
        self.config['DEFAULT'] = strconfig
        
        with open(self.config_file,'w') as f:
            self.config.write(f)
        
        logger.info('   ini file updated    ')
        
    def chimera_weight_update(self):
        # TODO : progressively update the loss weight between tasks
        
        # TODO : 1. scale the loss into the same magnitude
        
        # TODO : 2. update the weight by their own learning progress
        
        return None