import os
import torch
import torch.nn as nn
import pandas as pd
import yaml
import importlib.util as imp
from torch.utils.data import DataLoader
from data import dataset
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime

####################### QUESTE VARIABILI IDENTIFICATIVE ###################
idx = ['name', 
       'past_step', 
       'future_step', 
       'dropout',
       'num_layer_gnn_past', 
       'num_layer_gnn_future']
############################################################################

def step(model, 
         dataloader: DataLoader)-> pd.DataFrame:
    
    # put model in evaluation mode
    model.eval()
    yh = []
    real = []
    with torch.no_grad():
        for x_past, x_fut, y, adj in tqdm(iter(dataloader)):
            yh.append(model(x_past.to(model.device).float(), x_fut.to(model.device).float(), adj[0].to(model.device).float()).cpu().detach())
            real.append(y)
    yh = torch.cat(yh)
    real = torch.cat(real)
    err = yh-real
    loss = {
        'l1': [F.l1_loss(yh, real).item()],
        'MSE': [F.mse_loss(yh, real).item()], 
        'Infty': [torch.max(torch.abs(err)).item()], 
        'max_std':[torch.max(torch.std(err, 0)).item()],
        'expected_std':[torch.mean(torch.std(err, 0)).item()]
    }
    return pd.DataFrame.from_dict(loss)


def get_metric(config_env: yaml, 
               PATH:str, 
               ds: dataset):
    '''
    config_env (yaml): is the global configuration file
    PATH (str): is the path to the saved model
    ds (dataset): test dataset on which all the models are tested
    loss_function: global loss function on which testing the models
    '''
    name = config_env['setting']['name_model']

    ############### CARICO LA RELATIVA CLASSE ######################
    file_path = os.path.join(config_env['paths']['list_models'], name, 'model.py')
    # Create a module spec
    spec = imp.spec_from_file_location('model.py', file_path)
    # Import the module
    module = imp.module_from_spec(spec)
    spec.loader.exec_module(module)

    ############### INSTANZIO LA CLASSE ######################
    model = getattr(module, name)
    

    ######################
    id_model = name
    with open(os.path.join(config_env['paths']['config'], f"{id_model}.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    config.update(config_env)
    if 'dropout' in config_env['setting'].keys():
        config['model']['dropout'] = config_env['setting']['dropout']

    ############### Dataloader #####################
    dl_test = DataLoader(dataset = ds, 
                         batch_size = config['training']['batch_size'], 
                         shuffle = True)
    ################################################

    ################ Model #########################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model(in_feat_past = config['dataset']['in_feat_past'],
                in_feat_fut = config['dataset']['in_feat_future'],
                past = config['setting']['past_step'],
                future = config['setting']['future_step'],
                dropout = config['model']['dropout'],
                num_layer_gnn_past = config['model']['num_layer_gnn_past'],
                num_layer_gnn_future = config['model']['num_layer_gnn_future'],
                categorical_past = config['categorical'][config['setting']['dataset']]['past'],
                categorical_future = config['categorical'][config['setting']['dataset']]['future'],
                device = device).to(device)
    
    model.load_state_dict(torch.load(PATH))
    tmp = step(model = model, 
                dataloader = dl_test)
    tmp['num_layer_gnn_past'] = config['model']['num_layer_gnn_past'],
    tmp['num_layer_gnn_future'] = config['model']['num_layer_gnn_future']
    tmp['dropout'] = config['model']['dropout']
    tmp['past_step']= config['setting']['past_step']
    tmp['future_step']= config['setting']['future_step']
    tmp['data'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return tmp


def compare(config_env:yaml, 
            ds: dataset):
    
    past_step = config_env['setting']['past_step']
    future_step = config_env['setting']['future_step']
    exists = os.path.exists("./compare.csv")
    if exists:
        out = pd.read_csv("./compare.csv", index_col = 0)
    else:
        out = pd.DataFrame()
    loss = []
    for name in os.listdir(config_env['paths']['list_models']):
        print(name)
        tmp = f"{name}_{past_step}_{future_step}.pt"
        config_env['setting']['name_model']= name
        PATH = os.path.join(config_env['paths']['models'], tmp)
        if os.path.exists(PATH):
            tmp = get_metric(config_env=config_env, 
                             PATH=PATH, 
                             ds=ds)
            tmp['name']=name
            append = True

            if exists:
                row = out[idx].isin(tmp[idx].to_dict(orient = 'list')).all(axis =1)
                err = out[row]
                if len(err)>0:
                    if err.l1.values[0]>=tmp.l1.values[0]:
                        print("something better")
                        # poiché c'è stato un miglioramento rispetto alla norma desiderata
                        # si sostituisce tale valore all'interno del dataframe
                        out.iloc[out[row].index[out[row].index[0]]] = tmp.loc[0]
                    append = False
            if append:
                # La nuova riga del dataframe va appesa in due casi:
                # 1) il file csv non esiste
                # 2) quel modello non è mai stato valutato
                loss.append(tmp)
                print("append")
            torch.cuda.empty_cache()
    if exists:
        if len(loss)>0:
            loss = pd.concat(loss)
            out = pd.concat([out, loss])
    else:
        out = pd.concat(loss) 
    out = out.sort_values (by = 'l1')
    out.reset_index(drop=True, inplace=True)
    out.to_csv("./compare.csv")
    print(out)