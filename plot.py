import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import yaml
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

def plot(model, 
         config:yaml,
         loss_training: list, 
         loss_validation: list, 
         name:str,
         dl_train: DataLoader,
         dl_val: DataLoader, 
         show = False):

    fig = px.line({"epochs": range(1,len(loss_training)+1), 
                                   "train": loss_training, 
                                   "validation": loss_validation}, 
                                  x = "epochs", 
                                  y = ["train", "validation"], 
                                  title= f"training loss for {name}")
    fig.add_vline(x = np.argsort(loss_validation)[0]+1)
    fig.add_hline(y = np.min(loss_validation))
    fig.write_html(os.path.join(config['paths']['fig'], config['setting']['dataset'], f"loss_gnn_{name}.html"))
    if show:
        fig.show()
    model = model.cpu()
    model.device = torch.device('cpu')
    
    batch_train, y_train, adj_train = next(iter(dl_train))
    batch_val, y_val, adj_val = next(iter(dl_val))
    yh_train = model(batch_train.float().to(model.device), adj_train[0].to(model.device)).detach().numpy()
    yh_val = model(batch_val.float().to(model.device), adj_val[0].to(model.device)).detach().numpy()

    fig, ax = plt.subplots(nrows = y_val.shape[1], 
                           ncols = 2, 
                           constrained_layout = True,
                           figsize = (20, 3*y_val.shape[1]))
    
    for day in range(y_val.shape[1]):
        ax[day, 0].plot(yh_train[0,day], label = "estimate")
        ax[day, 0].plot(y_train[0,day], label ="real")
    
        ax[day, 1].plot(yh_val[0,day], label = "estimate")
        ax[day, 1].plot(y_val[0,day], label ="real")
        ax[day, 0].legend()
        ax[day, 1].legend()
    
        ax[day, 0].title.set_text(f"day {day +1} train")
        ax[day, 1].title.set_text(f"day {day +1} validation")
    fig.suptitle(' Comparison between estimation and reality ', fontsize=20) 
    
    path = os.path.join(config['paths']['fig'],config['setting']['dataset'], f"{name}.png")
    plt.savefig(path)
    if show:
        plt.show()
    plt.close(fig)

def plot_stream(model, 
                data: pd.DataFrame, 
                config: yaml, 
                adj:torch.tensor,
                n_nodes:int, 
                name:str, 
                timedelta: str = 'D'):
    past_step = model.past
    future_step = model.future
    date = np.sort(data.data.unique())
    y = []
    yh = []
    start = 0
    dt = np.diff(date[:past_step+future_step]) == np.timedelta64(1, timedelta)
    while any(not x for x in dt):
        start +=1
        dt = np.diff(date[start:past_step+future_step+ start]) == np.timedelta64(1, timedelta)
    
    for i in tqdm(range(start, len(date)-future_step-past_step-1)):
        if date[i+past_step+future_step]-date[i+past_step+future_step-1] == np.timedelta64(1, timedelta): 
            x = data[data.data.isin(date[i:i+past_step])].drop(columns = 'data').values
            x = x.reshape(1, past_step, n_nodes, config['setting']['in_feat'])
            tmp = data[data.data.isin(date[i+past_step:i+past_step+future_step])].y.values
            y.append(tmp.reshape(1, future_step, n_nodes))
            yh.append(F.relu(model(torch.from_numpy(x).to(model.device), adj[0].to(model.device)).detach()).cpu())
        else:
             i += past_step+future_step 
    
    yh = torch.cat(yh,0)
    y = np.vstack(y) 
    f = y.shape[-1]
    for step in range(model.future):
        fig, ax = plt.subplots(nrows = f, 
                                   ncols = 1, 
                                   constrained_layout = True,
                                   figsize = (20,f*3))
    
        for n in range(f):
            ax[n].plot(y[:,step,n], label = 'real')
            ax[n].plot(yh[:,step, n], label = 'estimated')
            ax[n].legend()  

            err = np.mean(np.abs(yh[:,step, n].numpy()-y[:,step, n]))
            ax[n].title.set_text(f"node {n}, step {step} train, err = {err}")
        
        path = os.path.join(config['paths']['fig'], config['setting']['dataset'], 'flows', name)
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, f"step{step+1}.png"))
        plt.close(fig)