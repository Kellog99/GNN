import yaml
import pickle
import os
from data import covid_dataset
import pandas as pd
import torch
import torch.nn.functional as F
from models.GLSTMseq2seq.main import model_GLSTMseq2seq
from models.GLSTM.main import model_GLSTM
from models.GAT_LSTM.main import model_GAT_LSTM


def get_dataloader(config:yaml)-> (covid_dataset, covid_dataset):
    past_step = config['setting']['past_step']
    future_step = config['setting']['future_step']

    with open( os.path.join(config['paths']['data'],f"aggregate/dataset{past_step}_{future_step}.pkl"), 'rb') as f:
        dataset = pickle.load(f) 

    len_train = int(len(dataset)*0.75)
    len_val = len(dataset)-len_train
    df_train, df_val = torch.utils.data.random_split(dataset=dataset, lengths = [len_train, len_val])
    return df_train, df_val

def linfty(y, yh, alpha):
        out = F.l1_loss(y,yh, reduction = "mean")
        out += alpha*torch.max(torch.abs(y-yh))
        return out

if __name__ == "__main__":
    with open("./config_env.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    def loss_function(y, yh):
         return linfty(y, yh, alpha = float(config['setting']['alpha']))
    
    df_train, df_val = get_dataloader(config = config)
    scores = {}
    
    be = model_GAT_LSTM(df_train = df_train, 
                            df_val = df_val, 
                            config_env = config,
                            loss_function = loss_function)
    scores['GAT_LSTM'] = be

    torch.cuda.empty_cache()                

    config['setting']['epochs'] = 1000
    be = model_GLSTMseq2seq(df_train = df_train, 
                            df_val = df_val, 
                            config_env = config,
                            loss_function = loss_function)
    scores['GLSTMseq2seq'] = be
    
    # it allows to delete all the objects that were stored into the GPU
    # in this way the memory of the GPU is cleared
    torch.cuda.empty_cache()                

    be = model_GLSTM(df_train = df_train, 
                    df_val = df_val, 
                    config_env = config,
                    loss_function = loss_function)
    scores['GLSTM'] = be
    print("Finish")

    scores = pd.DataFrame(scores.items(), columns = ['model', 'score'])
    scores.to_csv("./scores.csv")
    print(scores)