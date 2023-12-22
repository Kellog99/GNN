import yaml
import pickle
import os
from data import dataset
import pandas as pd
import torch
import torch.nn.functional as F
import importlib.util as imp
from functools import partial


#################### LOADING THE DATASET #####################
def get_dataloader(config:yaml)-> (dataset, dataset):
    past_step = config['setting']['past_step']
    future_step = config['setting']['future_step']

    with open( os.path.join(config['paths']['data'],f"dataset{past_step}_{future_step}.pkl"), 'rb') as f:
        dataset = pickle.load(f) 

    config['setting']['in_feat'] =  dataset[0][0].shape[-1]
    len_train = int(len(dataset)*0.75)
    len_val = len(dataset)-len_train
    df_train, df_val = torch.utils.data.random_split(dataset=dataset, lengths = [len_train, len_val])
    return df_train, df_val

#################### DEFINING THE LOSS FUNCTION #####################
def linfty(y:float, 
           yh:float, 
           alpha:float):
        out = F.l1_loss(y,yh, reduction = "mean")
        out += alpha*torch.max(torch.abs(y-yh))
        return out

#################### MAIN FUNCTION #####################
if __name__ == "__main__":
    with open("./config_env.yaml", 'r') as f:
        config = yaml.safe_load(f)

    #Set the weights of the piece of the loss
    loss_function = partial(linfty, alpha = float(config['setting']['alpha']))
    df_train, df_val = get_dataloader(config = config)
    scores = {}

    for model in os.listdir(config['paths']['list_models']):
         # mi inserisco all'interno del folder di ciascun modello
         if model == 'GCN_LSTM':
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
            scores[f"{model}"] = be
    scores = pd.DataFrame(scores.items(), columns = ['model', 'score'])
    scores.to_csv("./scores.csv")
    print(scores)