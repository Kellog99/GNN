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

    with open( os.path.join(config['paths']['data'], config['setting']['dataset'],f"{past_step}_{future_step}.pkl"), 'rb') as f:
        dataset = pickle.load(f) 
    
    config['setting']['in_feat'] =  dataset.x[0].shape[-1]
    len_train = int(len(dataset)*0.75)
    len_val = len(dataset)-len_train
    df_train, df_val = torch.utils.data.random_split(dataset=dataset, lengths = [len_train, len_val])
    return df_train, df_val

#################### DEFINING THE LOSS FUNCTION #####################
def linfty(y, yh, gamma = 1.0, alpha=1e-1, beta=1e-0):
    out = 0.5*F.l1_loss(y,yh, reduction = "mean")
    out += gamma*F.mse_loss(y,yh, reduction = "mean")
    out += alpha*torch.max(torch.abs(y-yh))
    out += beta*(torch.sum(F.relu(-yh)))
    return out

#################### MAIN FUNCTION #####################
if __name__ == "__main__":
    with open("./config_env.yaml", 'r') as f:
        config = yaml.safe_load(f)
    txt = f" tesing on {config['setting']['dataset']} "
    x = txt.center(80, "#")
    print(x) 
    ############### LOSS FUNCTION ######################
    loss_function = partial(linfty, alpha = float(config['setting']['alpha']))
    ####################################################
    
    ############### DATALOADER #########################
    df_train, df_val = get_dataloader(config = config)
    ####################################################

    scores = {}
    print(os.listdir(config['paths']['list_models']))
    for model in ['GCN_LSTM','GLSTM', 'GLSTMseq2seq',  'GAT_LSTMseq2seq', 'GAT_LSTM']:#os.listdir(config['paths']['list_models']):
        # mi inserisco all'interno del folder di ciascun modello
        ################## IMPORTING THE FUNCTION ####################
        file_path = os.path.join(config['paths']['list_models'], model, 'main.py')
        print(model)
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

        # it allows to delete all the objects that were stored into the GPU
        # in this way the memory of the GPU is cleared
        torch.cuda.empty_cache()      

    scores = pd.DataFrame(scores.items(), columns = ['model', 'score'])
    scores.to_csv("./scores.csv")
    print(scores) 