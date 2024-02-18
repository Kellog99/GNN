import yaml
import pickle
import os
from data import dataset
import torch
import pandas as pd
import torch.nn.functional as F
import importlib.util as imp
from functools import partial
from compare import check_update, get_metric

#################### LOADING THE DATASET #####################
def get_dataloader(config:yaml)-> tuple[dataset, dataset, dataset]:
    past_step = config['setting']['past_step']
    future_step = config['setting']['future_step']

    PATH = os.path.join(config['paths']['data'], config['setting']['dataset'],f"{past_step}_{future_step}.pkl")
    if os.path.exists(PATH):
        with open( PATH, 'rb') as f:
            dataset = pickle.load(f) 
    else:
        with open(os.path.join(config['paths']['adj'],"adj_totale.pkl"), "rb") as f:
            adj = pickle.load(f)
        data = pd.read_csv(os.path.join(config['paths']['data'], 'covid.csv'), index_col=0)
        data.data = pd.to_datetime(data.data, format="%Y-%m-%d")
        data.rename(columns = {'nuovi_casi':'y'}, inplace=True)

        PATH = os.path.join(config['paths']['data'], config['setting']['dataset'], f"{past_step}_{future_step}.pkl")
        dataset = dataset(df = data, 
                            past_step = past_step,
                            future_step = future_step, 
                            nodes = len(data.codice_provincia.unique()),
                            future_variables = data.columns[-7:].tolist(), 
                            y = ['y'],
                            adj = adj, 
                            timedelta = 'D')
    
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


def call_model(config:yaml, 
               df_train: dataset,
               df_val: dataset, 
               df_test: dataset):
    
    ############### LOSS FUNCTION ######################
    loss_function = partial(linfty, 
                            alpha = float(config['loss_function']['alpha']),
                            beta = float(config['loss_function']['beta']),
                            gamma = float(config['loss_function']['gamma']),
                            delta = float(config['loss_function']['delta']))
    ####################################################

    # mi inserisco all'interno del folder di ciascun modello
    ################## IMPORTING THE FUNCTION ####################
    file_path = os.path.join(config['paths']['list_models'], config['setting']['name_model'], 'main.py')
    # Create a module spec
    spec = imp.spec_from_file_location('main.py', file_path)
    # Import the module
    module = imp.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ = module.get_model(df_train = df_train, 
                        df_val = df_val, 
                        config_env = config,
                        loss_function = loss_function)
    torch.cuda.empty_cache()  
    
    ############# TESTING ################
    PATH = os.path.join(config['paths']['models'], f"{config['setting']['name_model']}_{config['setting']['past_step']}_{config['setting']['future_step']}.pt")

    tmp = get_metric(config_env=config, 
                    PATH=PATH,
                    ds=df_test)
    tmp['name']=config['setting']['name_model']
    check_update(tmp)

#################### TRAINING THE MODELS #####################
def ranking_list(config):
    ############### DATALOADER #########################
    df_train, df_val, df_test = get_dataloader(config = config)
    ####################################################

    #list_models = [x for x in list(os.listdir(config['paths']['list_models'])) if "seq2seq" in x]
    list_models = ["GAT_LSTMseq2seq"]
    list_models.sort()
    if bool(config['setting']['train']):
        print("models to be tested:")
        print('\n'.join([ f"* {x}" for x in list_models]))
        
        for model in list_models:
            txt = f" {model} ".center(40, "#")
            print(txt) 

            config['setting']['name_model'] = model
            call_model(config = config, 
                       df_train=df_train,
                       df_val=df_val, 
                       df_test = df_test)
       
#################### MAIN FUNCTION #####################
if __name__ == "__main__":
    ############### LOAD CONFIG ######################
    with open("./config_env.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    ranking_list(config=config)

    

    
