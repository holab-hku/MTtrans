import torch
import os
import PATH
import copy
import utils
import numpy as np
from tqdm import tqdm
from models import reader
from models.popen import Auto_popen
from sklearn.metrics import r2_score
from scipy import stats

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

global postproc_mean_std

postproc_mean_std = {
    "MPA_U": {"mean":  6.532835230673117, "std":  1.577417132818392},
    "MPA_H": {"mean":  5.786840247447307, "std":  1.5859451175340078},
    "MPA_V": {"mean":  5.269930466671135,  "std":  1.355184160931213},
    "RP_293T" : {"mean":  -0.615189054995973, "std":  1.060197697174912},
    "RP_muscle": {"mean":  -0.21022818608263194, "std":  1.4772718609075497},
    "RP_PC3" : {"mean":  -0.33492097824717426, "std":  0.9745174360808637}
}

def standardize_by(x, mean_std_dict):
    # use the string
    if type(mean_std_dict)==str:
        assert mean_std_dict in postproc_mean_std.keys(), f'invalid task name, shoudld be {postproc_mean_std.keys()}'
        mean_std_dict = postproc_mean_std[mean_std_dict]
        
    m = mean_std_dict['mean']
    std = mean_std_dict['std']
    return (x - m) /std

def reverse_transform(x, mean_std_dict):
    # use the string
    if type(mean_std_dict)==str:
        assert mean_std_dict in postproc_mean_std.keys(), f'invalid task name, shoudld be {postproc_mean_std.keys()}'
        mean_std_dict = postproc_mean_std[mean_std_dict]
        
    m = mean_std_dict['mean']
    std = mean_std_dict['std']
    return x * std + mean

def reload_model(config_file, device, kfold_index):
    config_file = os.path.join(utils.script_dir, config_file)
    
    POPEN  = Auto_popen(config_file)
    if kfold_index is None:
        check_point = torch.load(POPEN.vae_pth_path, map_location=device)
    else:
        check_point = torch.load(
            POPEN.vae_pth_path.replace('.pth', '_cv%d.pth'%kfold_index),
            map_location=device)
        
    model = check_point['state_dict'].to(device)
    return model

def val_a_epoch(model, dataloader, device):
    y_true_ls = []
    y_pred_ls = []
    with_uAUG = []
    model.eval()
    label_uAUG=False
    with torch.no_grad():
        for X,Y in tqdm(dataloader):
            if Y.shape[1] ==2:
                label_uAUG = True
            X = X.float().to(device)
            y_true_ls.append(Y.cpu().numpy())

            y_pred = model(X)
            y_pred_ls.append(y_pred.cpu().numpy())
    
    y_pred_f = np.concatenate( y_pred_ls).flatten()
    
    
    if label_uAUG:
        with_uAUG = np.concatenate( [y[:,1] for y in y_true_ls]).flatten()
        y_true_f = np.concatenate( [y[:,0] for y in y_true_ls]).flatten()
    else:
        y_true_f = np.concatenate( y_true_ls).flatten()
    return y_true_f, y_pred_f, with_uAUG

def kfold_load_data(config_file, device, kfold_index, label_uAUG=False ,expand_MAP=False):
    config_file = os.path.join(utils.script_dir, config_file)
    assert os.path.exists(config_file)
    POPEN = Auto_popen(config_file)
    POPEN.kfold_index = kfold_index
    
    if expand_MAP:
        if ('human' in POPEN.cycle_set) or ('SubHuman' in POPEN.cycle_set):
            POPEN.cycle_set = set(POPEN.cycle_set+['vleng','unmod1'])
    
    loader_set = {}
    base_path = copy.copy(POPEN.split_like)
    base_csv = copy.copy(POPEN.csv_path)
    
    for subset in POPEN.cycle_set:
        if (subset in ['MPA_U', 'MPA_H', 'MPA_V', 'SubMPA_H']):
            datapopen = Auto_popen(os.path.join(utils.script_dir, 'log/Backbone/RL_hard_share/3M/schedule_lr.ini'))
            datapopen.split_like = [path.replace('cycle', subset) for path in base_path]
            datapopen.kfold_index = kfold_index
            datapopen.shuffle = False
            if label_uAUG:
                datapopen.aux_task_columns += ['with_uAUG']
            datapopen.other_input_columns = POPEN.other_input_columns
            datapopen.n_covar = POPEN.n_covar

        elif (subset in ['RP_293T', 'RP_muscle', 'RP_PC3']):
            datapopen = Auto_popen(os.path.join(utils.script_dir,'log/Backbone/RL_hard_share/3R/schedule_MTL.ini'))
            datapopen.csv_path = base_csv.replace("cycle",subset)
            datapopen.kfold_index = kfold_index
            datapopen.shuffle = False
            if label_uAUG:
                datapopen.aux_task_columns += ['with_uAUG']
            datapopen.other_input_columns = POPEN.other_input_columns
            datapopen.n_covar = POPEN.n_covar

        loader_set[subset] = reader.get_dataloader(datapopen)

    return loader_set

def pipeline_for_RL_3_data(config_file, which_set=1, device='cpu', label_uAUG=False, expand_MAP=False):
    
    device = torch.device('cuda:%s'%device) if type(device)==int else 'cpu'
    
    model = reload_model(config_file, device)
    
    r2_dict = {}
    for cycle in ['unmod1', 'human', 'vleng']:
        model.task = cycle # which works for normal RL_gru
        dataloader = loader_set[cycle][which_set]
        y_true_f, y_pred_f, with_uAUG = val_a_epoch(model, dataloader, device)
        r2_dict[cycle] = compute_r2(y_true_f,y_pred_f) , r2_score(y_true_f,y_pred_f)
    
    del model
    
    return r2_dict

def pipeline_for_kfold(config_file, which_set=1, device='cpu', kfold_index=None, label_uAUG=False, expand_MAP=False):
    
    device = torch.device('cuda:%s'%device) if type(device)==int else 'cpu'
    
    loader_set = kfold_load_data("log/Backbone/RL_hard_share/3M/small_repective_filed_strides1113.ini", device, kfold_index, label_uAUG)
    model = reload_model(config_file, device, kfold_index)
    
    r2_dict = {}
    ture_pred_dict = {}
    for cycle in list(loader_set.keys()):
        model.task = cycle # which works for normal RL_gru
        dataloader = loader_set[cycle][which_set]
        y_true_f, y_pred_f, with_uAUG = val_a_epoch(model, dataloader, device)
        r2_dict[cycle] = r2_score(y_true_f,y_pred_f)
        ture_pred_dict[cycle] = [y_true_f, y_pred_f]
        
    del model
    
    return r2_dict,ture_pred_dict

def pipeline_for_RL_gru(config_file, which_set=2, device='cpu', kfold_index=None):
    config_file = os.path.join("/ssd/users/wergillius/Project/UTR_VAE/", config_file)
    POPEN = Auto_popen(config_file)
    POPEN.kfold_index = kfold_index
    POPEN.shuffle = False
    dataloader = reader.get_dataloader(POPEN)[which_set]
    
    device = torch.device('cuda:%s'%device) if type(device)==int else 'cpu'
    
    model = reload_model(config_file, device, kfold_index)
    
    y_true_f, y_pred_f, with_uaug = val_a_epoch(model, dataloader, device)
    r2 =  r2_score(y_true_f,y_pred_f)

    del model
    
    return r2, (y_true_f,y_pred_f)