import yaml
import pickle
import os
from data import dataset
import torch
import torch.nn.functional as F
import importlib.util as imp
from functools import partial
from compare import compare

#################### LOADING THE DATASET #####################
def get_dataloader(config:yaml)-> (dataset, dataset, dataset):
    past_step = config['setting']['past_step']
    future_step = config['setting']['future_step']

    with open( os.path.join(config['paths']['data'], config['setting']['dataset'],f"{past_step}_{future_step}.pkl"), 'rb') as f:
        dataset = pickle.load(f) 
    
    config['dataset']['in_feat_past'] =  dataset.x[0].shape[-1]
    config['dataset']['in_feat_future'] =  dataset.x_fut[0].shape[-1]
    len_train = int(len(dataset)*config['dataset']['train_split'])
    len_val = int(len(dataset)*config['dataset']['val_split'])
    len_test = len(dataset)-(len_train+len_val)
    df_train, df_val, df_test = torch.utils.data.random_split(dataset=dataset, lengths = [len_train, len_val, len_test])
    
    return df_train, df_val, df_test

#################### DEFINING THE LOSS FUNCTION #####################
def linfty(y, yh, 
           alpha:float = 1e-1, 
           beta:float = 1e-0,
           gamma:float = 1.0, 
           delta:float = 1.0):
    
    out = alpha*F.l1_loss(y,yh, reduction = "mean")
    out += beta*F.mse_loss(y,yh, reduction = "mean")
    out += gamma*torch.max(torch.abs(y-yh))
    out += delta*(torch.sum(F.relu(-yh)))
    return out

#################### MAIN FUNCTION #####################
if __name__ == "__main__":
    ############### LOAD CONFIG ######################
    with open("./config_env.yaml", 'r') as f:
        config = yaml.safe_load(f)
    txt = f" testing on {config['setting']['dataset']} "
    x = txt.center(80, "#")
    print(x) 

    ############### LOSS FUNCTION ######################
    loss_function = partial(linfty, 
                            alpha = float(config['loss_function']['alpha']),
                            beta = float(config['loss_function']['beta']),
                            gamma = float(config['loss_function']['gamma']),
                            delta = float(config['loss_function']['delta']))
    ####################################################
    
    ############### DATALOADER #########################
    df_train, df_val, df_test = get_dataloader(config = config)
    ####################################################
    if bool(config['setting']['train']):
        for model in [ 'GCN_LSTM',  'GAT_LSTM', 'GAT_LSTMseq2seq', 'GLSTM', 'GLSTMseq2seq']:#os.listdir(config['paths']['list_models']):
            txt = f" {model} ".center(40, "#")
            print(txt) 

            config['setting']['name_model'] = model

            # mi inserisco all'interno del folder di ciascun modello
            ################## IMPORTING THE FUNCTION ####################
            file_path = os.path.join(config['paths']['list_models'], model, 'main.py')
            # Create a module spec
            spec = imp.spec_from_file_location('main.py', file_path)
            # Import the module
            module = imp.module_from_spec(spec)
            spec.loader.exec_module(module)
            be = module.get_model(df_train = df_train, 
                                df_val = df_val, 
                                config_env = config,
                                loss_function = loss_function)
            torch.cuda.empty_cache()  

    txt = f" comparing the results "
    x = txt.center(80, "#")
    print(x) 
    compare(config_env = config, 
            ds = df_test)