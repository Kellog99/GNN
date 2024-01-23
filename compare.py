import os
import torch
import torch.nn as nn
import pandas as pd
import yaml
import importlib.util as imp
from torch.utils.data import DataLoader
from data import dataset
from tqdm import tqdm

def step(model, 
         dataloader: DataLoader, 
         loss_function: callable)-> float:
    # put model in evaluation mode
    model.eval()
    loss_step = 0
    with torch.no_grad():
        for x_past, x_fut, y, adj in tqdm(iter(dataloader)):
            yh = model(x_past.to(model.device).float(), x_fut.to(model.device).float(), adj[0].to(model.device).float())
            loss = loss_function(yh, y.to(model.device).float())
            loss_step += loss.item()
    loss_step = loss_step/len(dataloader)
    return loss_step

def compare(config_env:yaml, 
            ds: dataset, 
            loss_function:callable = nn.L1Loss(reduction="mean")):
    
    past_step = config_env['setting']['past_step']
    future_step = config_env['setting']['future_step']
    
    loss = {}
    for name in os.listdir(config_env['paths']['list_models']):
        tmp = f"{name}_{past_step}_{future_step}.pt"
        PATH = os.path.join(config_env['paths']['models'], tmp)
        if os.path.exists(PATH):
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
            

            ############### Characterization ###############
            batch_size= config['training']['batch_size']
            past_step = config['setting']['past_step']
            future_step = config['setting']['future_step']
            ################################################

            ############### Dataloader #####################
            dl_test = DataLoader(dataset = ds, batch_size = batch_size, shuffle = True)
            ################################################

            ################ Model #########################
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model(in_feat_past = config['dataset']['in_feat_past'],
                        in_feat_fut = config['dataset']['in_feat_future'],
                        past = past_step,
                        future = future_step,
                        dropout = config['model']['dropout'],
                        categorical_past = config['categorical'][config['setting']['dataset']]['past'],
                        categorical_future = config['categorical'][config['setting']['dataset']]['future'],
                        device = device).to(device)
            tmp = step(model = model, 
                       dataloader = dl_test,
                       loss_function = loss_function)
            loss[name] = tmp
    loss = pd.DataFrame(loss.items(), columns = ['model', 'score']).sort_values(by = 'score').reset_index(drop = True)
    loss['past_step']= past_step
    loss['future_step']= future_step
    loss.to_csv("./compare.csv")
    print(loss)