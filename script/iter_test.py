import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

parser = argparse.ArgumentParser('the main to train model')
parser.add_argument('--config_file',type=str,required=True)
parser.add_argument('--cuda',type=str,default=0,required=False)
parser.add_argument("--kfold_index",type=int,default=1,required=False)
args = parser.parse_args()

cuda_id = args.cuda if args.cuda is not None else utils.get_config_cuda(args.config_file)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)


import time
import torch
import copy
import utils
from torch import optim
import numpy as np 
from models import reader,train_val
from models.ScheduleOptimizer import ScheduledOptim,scheduleoptim_dict_str
from models.popen import Auto_popen
from models.loss import Dynamic_Task_Priority,Dynamic_Weight_Averaging


POPEN = Auto_popen(args.config_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POPEN.cuda_id = device


POPEN.kfold_index = args.kfold_index
if POPEN.kfold_cv:
    if args.kfold_index is None:
        raise NotImplementedError("please specify the kfold index to perform K fold cross validation")
    POPEN.vae_log_path = POPEN.vae_log_path.replace(".log","_cv%d.log"%args.kfold_index)
    POPEN.vae_pth_path = POPEN.vae_pth_path.replace(".pth","_cv%d.pth"%args.kfold_index)
    

# Run name
if POPEN.run_name is None:
    run_name = POPEN.model_type + time.strftime("__%Y_%m_%d_%H:%M")
else:
    run_name = POPEN.run_name
    
# log dir
logger = utils.setup_logs(POPEN.vae_log_path)
logger.info(f"    ===========================| device {device}{cuda_id} |===========================    ")
#  built model dir or check resume 
POPEN.check_experiment(logger)

#                               |=====================================|
#                               |===========   setup  part  ==========|
#                               |=====================================|
# read data
loader_set = {}                                                                                                                                                                                                                                                                                                                 
base_path = ['cycle_train_val.csv', 'cycle_test.csv']
base_csv = 'cycle_MTL_transfer.csv'
for subset in POPEN.cycle_set:
    if (subset in ['MPA_U', 'MPA_H', 'MPA_V', 'SubMPA_H']):
        datapopen = Auto_popen('log/Backbone/RL_hard_share/3M/schedule_lr.ini')
        datapopen.split_like = [path.replace('cycle', subset) for path in base_path]
        datapopen.kfold_index = args.kfold_index
        
        datapopen.other_input_columns = POPEN.other_input_columns
        datapopen.n_covar = POPEN.n_covar

    elif (subset in ['RP_293T', 'RP_muscle', 'RP_PC3', 'pcr3', '293']):
        datapopen = Auto_popen('log/Backbone/RL_hard_share/3R/schedule_MTL.ini')
        datapopen.csv_path = base_csv.replace("cycle",subset)
        datapopen.kfold_index = args.kfold_index

        datapopen.other_input_columns = POPEN.other_input_columns
        datapopen.n_covar = POPEN.n_covar

    loader_set[subset] = reader.get_dataloader(datapopen)

    
# ===========  setup model  ===========
# train_iter = iter(train_loader)
# X,Y  = next(train_iter)
# -- pretrain -- 
if POPEN.pretrain_pth is not None:
    # load pretran model
    logger.info("===============================|   pretrain   |===============================")
    logger.info(f" {POPEN.pretrain_pth}")
    pretrain_popen = Auto_popen(os.path.join(utils.script_dir, POPEN.pretrain_pth))
    pretrain_model = torch.load(pretrain_popen.vae_pth_path, map_location=torch.device('cpu'))['state_dict']

    
    if POPEN.model_type == pretrain_popen.model_type:
        # if not POPEN.Resumable:
        #     # we only load pre-train for the first time 
        #     # later we can resume 
        model = pretrain_model.to(device)
        del pretrain_model
        
        if (POPEN.cycle_set != pretrain_popen.cycle_set):
            model.all_tasks = POPEN.cycle_set
            model.tower = torch.nn.ModuleDict(
                {POPEN.cycle_set[i] : model.tower[t]  for i, t in enumerate(pretrain_popen.cycle_set)}
                                                )
        
    elif POPEN.modual_to_fix is not None:
        # POPEN.model_type != pretrain_popen.model_type
        model = POPEN.Model_Class(*POPEN.model_args)
        for modual in POPEN.modual_to_fix:
            if modual in dir(pretrain_model):    
                eval(f'model.{modual}').load_state_dict(
                    eval(f'model.{modual}').state_dict()
                    )
        model =  model.to(device)
        
    else:
        # two different class -> Enc_n_Down
        downstream_model = POPEN.Model_Class(*POPEN.model_args)
        # merge 
        model = MTL_models.Enc_n_Down(pretrain_model,downstream_model).to(device)
    
# -- end2end -- 
elif POPEN.model_type == "CrossStitch_Model":
    backbone = {}
    for t in POPEN.tasks:
        task_popen = Auto_popen(POPEN.backbone_config[t])
        task_model = task_popen.Model_Class(*task_popen.model_args)
        utils.load_model(task_popen,task_model,logger)
        backbone[t] = task_model.to(device)
    POPEN.model_args = [backbone] + POPEN.model_args
    model = POPEN.Model_Class(*POPEN.model_args).to(device)
else:
    Model_Class = POPEN.Model_Class  # DL_models.LSTM_AE 
    model = Model_Class(*POPEN.model_args).to(device)
    
if POPEN.Resumable:
    model = utils.load_model(POPEN, model, logger)
    
# =========== fix parameters ===========
if isinstance(POPEN.modual_to_fix, list):
    for modual in POPEN.modual_to_fix:
        model = utils.fix_parameter(model,modual)
        model =  model.to(device)
    logger.info(' \t \t ==============| %s fixed |============== \t \t \n'%POPEN.modual_to_fix)
# =========== set optimizer ===========
if POPEN.optimizer == 'Schedule':
    optimizer = ScheduledOptim(optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                      betas=(0.9, 0.98), 
                                      eps=1e-09, 
                                      weight_decay=1e-4, 
                                      amsgrad=True),
                           n_warmup_steps=20)
elif type(POPEN.optimizer) == dict:
    optimizer = eval(scheduleoptim_dict_str.format(**POPEN.optimizer))
else:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=POPEN.lr,
                            betas=(0.9, 0.98), 
                            eps=1e-09, 
                            weight_decay=POPEN.l2)
if POPEN.loss_schema == 'DTP':
    POPEN.loss_schedualer = Dynamic_Task_Priority(POPEN.tasks,POPEN.gamma,POPEN.chimerla_weight)
elif POPEN.loss_schema == 'DWA':
    POPEN.loss_schedualer = Dynamic_Weight_Averaging(POPEN.tasks,POPEN.tau,POPEN.chimerla_weight)
# =========== resume ===========
best_loss = np.inf
best_acc = 0
best_epoch = 0
previous_epoch = 0
epoch = 0
if POPEN.Resumable:
    previous_epoch,best_loss,best_acc = utils.resume(POPEN, optimizer,logger)
    

#                               |=====================================|
#                               |==========  training  part ==========|
#                               |=====================================|
epoch += previous_epoch



#              -----------| validate |-----------   
logger.info("===============================| test |===============================")
verbose_dict = train_val.cycle_validate(loader_set,model,optimizer,popen=POPEN,epoch=epoch, which_set=2)
# matching task performance influence what to save

if np.any(['r2' in key for key in verbose_dict.keys()]):
    val_avg_acc = np.mean([values for key, values in verbose_dict.items() if 'r2' in key])
    acc_dict = {f"cv{args.kfold_index}_{key}":values for key, values in verbose_dict.items() if 'r2' in key}
else:
    val_avg_acc = np.mean([values for key, values in verbose_dict.items() if 'acc' in key])
    acc_dict = {}
val_total_loss = verbose_dict['Total']

DICT ={"ran_epoch":epoch,"n_current_steps":optimizer.n_current_steps,"delta":optimizer.delta} if type(optimizer) == ScheduledOptim else {"ran_epoch":epoch}
POPEN.update_ini_file(DICT,logger)
    
    
#    -----------| compare the result |-----------
if (best_loss > val_total_loss) :
    # update best performance
    best_loss = min(best_loss,val_total_loss)
    best_acc = max(best_acc,val_avg_acc)
    best_epoch = epoch
    
    # save
    utils.snapshot(POPEN.vae_pth_path, {
                'epoch': epoch + 1,
                'validation_acc': val_avg_acc,
                # 'state_dict': model.state_dict(),
                'state_dict': model,
                'validation_loss': val_total_loss,
                'optimizer': optimizer.state_dict(),
            })
    
    # update the popen
    # r2_dict = {f"cv{args.kfold_index}_{key}":values for key, values in verbose_dict.items() if 'r2' in key}
    to_update = {'run_name':run_name, "ran_epoch":epoch,"best_acc":best_acc}

    POPEN.update_ini_file(to_update.update(acc_dict),logger)