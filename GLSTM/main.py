import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import yaml
import pickle
from torch.utils.data import DataLoader
from model import GLSTM
from training import training
import sys
sys.path.insert(1,"/home/andrea/Scrivania/Tesi/leonardo/notebook")
from data import covid_dataset
sys.path.insert(1,"/home/andrea/Scrivania/Tesi/leonardo")
from plot import plot

if __name__ == "__main__":
    with open("./config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    past_step = config['setting']['past_step']
    future_step = config['setting']['future_step']
    batch_size= config['setting']['batch_size']

    with open( os.path.join(config['paths']['data'],f"aggregate/dataset{past_step}_{future_step}.pkl"), 'rb') as f:
        dataset = pickle.load(f)
    len_train = int(len(dataset)*0.75)
    len_val = len(dataset)-len_train
    df_train, df_val = torch.utils.data.random_split(dataset=dataset, lengths = [len_train, len_val])
    dl_train = DataLoader(dataset=df_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(dataset=df_val, batch_size=batch_size, shuffle=True)
    batch, y, adj = next(iter(dl_train))
    print(batch.shape)
    print(y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GLSTM(in_feat = batch.float().shape[-1], 
                past = past_step,
                future = future_step,
                categorical = config['model']['categorical'],
                embedding = config['model']['embedding'],
                device = device).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def linfty(y, yh, alpha = 7e-5):
        out = F.l1_loss(y,yh, reduction = "mean")
        #out += alpha*torch.max(torch.sum(torch.abs(y-yh), (-2,-1)))
        out += alpha*torch.max(torch.abs(y-yh))
        return out
    optimizer = optim.Adam(model.parameters(), 
                        lr = 1e-4, 
                        weight_decay = 2e-4)
    model.to(device)
    model.device = device
    model, loss_training, loss_validation = training(model, 
                                                train_loader = dl_train, 
                                                val_loader = dl_val, 
                                                num_epochs = config['setting']['epochs'], 
                                                criterion = linfty,
                                                optimizer = optimizer)
    
    torch.save(model.state_dict(), os.path.join(config['paths']['models'], f"GLSTM_{past_step}_{future_step}.pt"))

    plot(model,
        config,
        loss_training, 
        loss_validation, 
        dl_train=dl_train, 
        dl_val=dl_val, 
        name = f"GLSTM_{past_step}_{future_step}", 
        show = True)