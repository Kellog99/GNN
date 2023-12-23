import os
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from models.GAT_LSTM.model import GAT_LSTM
from plot import plot
from data import dataset
from trainer import Trainer

def get_model(df_train : dataset, 
                df_val : dataset, 
                config_env: yaml, 
                loss_function):

    id_model = "GAT-LSTM"
    with open(os.path.join(config_env['paths']['config'], f"{id_model}.yaml"), 'r') as f:
        config = yaml.safe_load(f)
        
    config.update(config_env)

    if 'epochs' in config_env['setting'].keys():
        config['training']['epochs'] = config_env['setting']['epochs']
    
    ############### Characterization ###############
    batch_size= config['training']['batch_size']
    past_step = config['setting']['past_step']
    future_step = config['setting']['future_step']
    id_test = f"{id_model}_{past_step}_{future_step}"
    ################################################

    ############### Dataloader #####################
    dl_train = DataLoader(dataset = df_train, batch_size = batch_size, shuffle = True)
    dl_val = DataLoader(dataset = df_val, batch_size = batch_size, shuffle = True)
    batch, _, _ = next(iter(dl_train))
    in_feat = batch.shape[-1]
    print(batch.shape)
    ################################################

    ################ Model #########################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT_LSTM(in_feat = in_feat, 
                past = past_step,
                future = future_step,
                categorical = config['categorical'][config['setting']['dataset']],
                device = device).to(device)
        
    optimizer = optim.Adam(model.parameters(), 
                           lr = float(config['setting']['lr']), 
                           weight_decay = float(config['setting']['weight_decay']))
    ################################################

     ################ Training ######################
    trainer = Trainer(model = model, 
                      PATH = os.path.join(config['paths']['models'], f"{id_test}.pt"), 
                      optimizer=optimizer, 
                      loss_function=loss_function)
    
    trainer.fit(train_loader = dl_train, 
                val_loader = dl_val, 
                epochs = config['training']['epochs'])    
    #################################################

    plot(model = trainer.model,
        config = config,
        loss_training = trainer.loss_train, 
        loss_validation = trainer.loss_val, 
        dl_train = dl_train, 
        dl_val = dl_val, 
        name = f"{id_test}")

    return min(trainer.loss_val)