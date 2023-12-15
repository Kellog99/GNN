import os
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from models.GAT_LSTM.model import GAT_LSTM
from models.GAT_LSTM.training import training
from plot import plot
from data import covid_dataset

def model_GAT_LSTM(df_train : covid_dataset, 
                df_val : covid_dataset, 
                config_env: yaml, 
                loss_function):

    with open("/home/andrea/Scrivania/Tesi/leonardo/models/GAT_LSTM/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    config.update(config_env)

    if 'epochs' in config_env['setting'].keys():
        config['training']['epochs'] = config_env['setting']['epochs']
    
    ############### Characterization ###############
    batch_size= config['training']['batch_size']
    past_step = config['setting']['past_step']
    future_step = config['setting']['future_step']
    id_model = f"GAT_LSTM_{past_step}_{future_step}"
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
                categorical = config['model']['categorical'],
                device = device).to(device)
        
    optimizer = optim.Adam(model.parameters(), 
                           lr = float(config['setting']['lr']), 
                           weight_decay = float(config['setting']['weight_decay']))
    ################################################

    ################ Training ######################
    model, loss_training, loss_validation = training(model, 
                                                    train_loader = dl_train, 
                                                    val_loader = dl_val, 
                                                    num_epochs = config['setting']['epochs'], 
                                                    criterion = loss_function,
                                                    optimizer = optimizer)
    
    torch.save(model.state_dict(), os.path.join(config['paths']['models'], f"{id_model}.pt"))
    #################################################

    plot(model,
        config,
        loss_training, 
        loss_validation, 
        dl_train=dl_train, 
        dl_val=dl_val, 
        name = f"{id_model}", 
        show = False)

    return min(loss_validation)