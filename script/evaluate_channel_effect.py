import os
import sys
import pandas as pd
import numpy as np
import PATH
import torch
import argparse
# 
from models import reader
from models import train_val
from models.popen import Auto_popen
from models import max_activation_patch as MAP
# 
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
import warnings
warnings.filterwarnings('ignore')


#
parser = argparse.ArgumentParser('the script to evlauate the effect of ')
parser.add_argument("-c", "--config", type=str, required=True, help='the model config file: xxx.ini')
parser.add_argument("-s", "--set", type=int, default=2, help='train - 0 ,val - 1, test - 2 ')
parser.add_argument("-p", "--n_max_act", type=int, default=500, help='the number of seq')
parser.add_argument("-k", "--kfold_cv", type=int, default=1, help='the repeat')
parser.add_argument("-d", "--device", type=str, default='cpu', help='the device to use to extract featmap, digit or cpu')
args = parser.parse_args()

config_path = args.config
save_path = config_path.replace(".ini", "_coef")
config = Auto_popen(config_path)
config.batch_size = 256
config.kfold_cv = 'train_val'
all_task = config.cycle_set

task_channel_effect = {}
task_performance = {}

# path check
if os.path.exists(config_path) and not os.path.exists(save_path):
    os.mkdir(save_path)

for task in all_task:

    # .... format featmap as data ....
    print(f"\n\nevaluating for task: {task}")
    # re-instance the map for each task
    map_task = MAP.Maximum_activation_patch(popen=config, which_layer=4,
                                      n_patch=args.n_max_act,
                                      kfold_index=args.kfold_cv,
                                      device_string=args.device)

    # extract feature map and rl decision chain
    featmap = map_task.extract_feature_map(task=task, which_set=args.set)
    cum_rl_trend = map_task.cumulative_rl_decision(task=task, which_set=args.set)

    # truncate the featmap and rl trend according to sequence length
    max_seq_len = map_task.df[config.seq_col].apply(len).max()
    to_stay = max_seq_len // np.product(map_task.strides) +1
    trunc_start = featmap.shape[2] - to_stay
    featmap = featmap[:,:,trunc_start:]
    cum_rl_trend = cum_rl_trend[:,trunc_start:]

    # construct input for linear regression
    n_sample,n_channel,n_posi = featmap.shape
    
    X = featmap.reshape(n_sample,-1)
    Y = map_task.Y_ls.flatten()

    # .... regression ....

    # Lasso Regression
    print("\nregressing Lasso..")
    L1 = LassoCV(alphas=np.linspace(2e-3, 0.1, 49)).fit(X,Y)
    L1_r2 = L1.score(X,Y)
    L1_sparsity = np.sum(L1.coef_==0) / X.shape[1] *100
    print(f"Lasso with optimal alpha {L1.alpha_:.5f}, r2 {L1_r2:.3f} , zero coeff {L1_sparsity:.1f}%")
    # save df
    L1_effect = L1.coef_.reshape(n_channel,-1)
    L1_fullcoef_df = pd.DataFrame(L1_effect, columns=["posi_"+str(trunc_start+i) for i in range(to_stay)])
    L1_fullcoef_df.to_csv( os.path.join(save_path , f"{task}_L1_coef.csv"), index=False)
    task_channel_effect[f"{task}_L1"] = L1_effect.mean(axis=1)


    # Ridge Regression
    print("\nregressing Ridge..")
    L2 = RidgeCV(alphas=np.linspace(0.001, 0.101, 20)).fit(X,Y)
    L2_r2 = L2.score(X,Y)
    L2_sparsity = np.sum(L2.coef_==0) / X.shape[1] *100
    print(f"Lasso with optimal alpha {L2.alpha_:.5f}, r2 {L2_r2:.3f} , zero coeff {L2_sparsity:.1f}%")
    # save df
    L2_effect = L2.coef_.reshape(n_channel,-1)
    L2_fullcoef_df = pd.DataFrame(L2_effect, columns=["posi_"+str(trunc_start+i) for i in range(to_stay)])
    L2_fullcoef_df.to_csv( os.path.join(save_path , f"{task}_L2_coef.csv"), index=False)
    task_channel_effect[f"{task}_L2"] = L2_effect.mean(axis=1)

    # ElasticNet Regression
    print("regressing ElasticNet..")
    elastic = ElasticNetCV(alphas=np.linspace(2e-3, 0.1, 49), n_jobs=10).fit(X, Y)   
    Ela_r2 = elastic.score(X,Y)
    Ela_sparsity = np.sum(elastic.coef_==0) / X.shape[1] *100
    print(f"Lasso with optimal alpha {elastic.alpha_:.5f}, r2 {Ela_r2:.3f} , zero coeff {Ela_sparsity:.1f}%")
    # save df
    elastic_effect = elastic.coef_.reshape(n_channel,-1)
    elastic_fullcoef_df = pd.DataFrame(elastic_effect, columns=["posi_"+str(trunc_start+i) for i in range(to_stay)])
    elastic_fullcoef_df.to_csv( os.path.join(save_path , f"{task}_Elastic_coef.csv"), index=False)
    task_channel_effect[f"{task}_Elastic"] = elastic_effect.mean(axis=1)


    task_performance[f"{task}_L1"] = [L1.alpha_, L1_r2, L1_sparsity, L1.coef_.max(), L1.coef_.min()]  
    task_performance[f"{task}_L2"] =  [L2.alpha_, L2_r2, L2_sparsity, L2.coef_.max(), L2.coef_.min()]
    task_performance[f"{task}_Elastic"] =  [elastic.alpha_, Ela_r2, Ela_sparsity, elastic.coef_.max(), elastic.coef_.min()]

all_effect = pd.DataFrame(task_channel_effect)
all_effect.to_csv(os.path.join(save_path, "all_task_mean_effect.csv"), index=False)

report_df = pd.DataFrame(task_performance)
report_df.index = ['optim_alpha','r2', 'zero_pctg', 'max_coef', 'min_coef']
report_df.to_csv(os.path.join(save_path, "regression_report.csv"), index=False)
