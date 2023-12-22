import os
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from models.GCN_LSTM.model import GCN_LSTM
from plot import plot
from data import covid_dataset
from trainer import Trainer


def get_model(df_train : covid_dataset, 
                df_val : covid_dataset, 
                config_env: yaml, 
                loss_function):

    id_model = "GCN-LSTM"
    with open(os.path.join(config_env['paths']['config'], f"{id_model}.yaml"), 'r') as f:
        config = yaml.safe_load(f)
        
    config.update(config_env)

    if 'epochs' in config_env['setting'].keys():
        config['training']['epochs'] = config_env['setting']['epochs']
    
    ############### Characterization ###############
    batch_size= config['training']['batch_size']
    past_step = config['setting']['past_step']
    future_step = config['setting']['future_step']
    id_model = f"{id_model}_{past_step}_{future_step}"
    ################################################

    ############### Dataloader #####################
    dl_train = DataLoader(dataset = df_train, batch_size = batch_size, shuffle = True)
    dl_val = DataLoader(dataset = df_val, batch_size = batch_size, shuffle = True)
    ################################################

    ################ Model #########################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN_LSTM(in_feat = config['setting']['in_feat'], 
                past = past_step,
                future = future_step,
                categorical = config['model']['categorical'],
                embedding = config['model']['embedding'],
                device = device).to(device)
        
    optimizer = optim.Adam(model.parameters(), 
                           lr = float(config['setting']['lr']), 
                           weight_decay = float(config['setting']['weight_decay']))
    ################################################

    ################ Training ######################
    trainer = Trainer(model = model, 
                      PATH = os.path.join(config['paths']['models'], f"{id_model}.pt"), 
                      optimizer=optimizer, 
                      loss_function=loss_function)
    
    trainer.fit(train_loader = dl_train, 
                val_loader = dl_val, 
                epochs = config['training']['epochs'])
    
    torch.save(model.state_dict(), os.path.join(config['paths']['models'], f"{id_model}.pt"))
    #################################################

    plot(trainer.model,
        config,
        trainer.loss_train, 
        trainer.loss_val, 
        dl_train = dl_train, 
        dl_val = dl_val, 
        name = f"{id_model}", 
        show = False)

    return min(trainer.loss_val)