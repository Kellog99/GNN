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
         show = False, 
         n_nodes:int=30):

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
    
    x_past_train, x_fut_train, y_train, adj_train = next(iter(dl_train))
    x_past_val, x_fut_val, y_val, adj_val = next(iter(dl_val))
    yh_train = model(x_past_train.float().to(model.device), x_fut_train.float().to(model.device), adj_train[0].to(model.device)).detach().numpy()
    yh_val = model(x_past_val.float().to(model.device), x_fut_val.float().to(model.device), adj_val[0].to(model.device)).detach().numpy()

    n_nodes = min(y_val.shape[1], n_nodes)
    fig, ax = plt.subplots(nrows = n_nodes, 
                           ncols = 2, 
                           constrained_layout = True,
                           figsize = (20, 3*n_nodes))
    
    for day in range(n_nodes):
        ax[day, 0].plot(yh_train[0,day], label = "estimate")
        ax[day, 0].plot(y_train[0,day], label ="real")
    
        ax[day, 1].plot(yh_val[0,day], label = "estimate")
        ax[day, 1].plot(y_val[0,day], label ="real")
        ax[day, 0].legend()
        ax[day, 1].legend()
    
        ax[day, 0].title.set_text(f"node {day +1} train")
        ax[day, 1].title.set_text(f"node {day +1} validation")
    fig.suptitle(' Comparison between estimation and reality ', fontsize=20) 
    
    path = os.path.join(config['paths']['fig'], config['setting']['dataset'], f"{name}.png")
    plt.savefig(path)
    if show:
        plt.show()
    plt.close(fig)

def plot_stream(model, 
                df: pd.DataFrame, 
                config: yaml, 
                past_variables:list, 
                future_variables: list,
                adj:torch.tensor,
                nodes:int, 
                name:str, 
                node: int = 3, 
                timedelta: str = 'D'):
    node = min(node, nodes)-1
    past_step = model.past
    future_step = model.future
    date = np.sort(df.data.unique())
    y = []
    yh = []
    start = 0

    x_past = df.groupby('data').apply(lambda x : np.array(x[past_variables].values))
    x_fut = df.groupby('data').apply(lambda x : np.array(x[future_variables].values))
    y_group = df.groupby('data').apply(lambda x : np.array(x['y'].values))
    print(y_group.shape)
    dt = np.diff(date[:past_step+future_step]) == np.timedelta64(1, timedelta)
    while any(not x for x in dt):
        start +=1
        dt = np.diff(date[start:past_step+future_step+ start]) == np.timedelta64(1, timedelta)
    
    for i in tqdm(range(start, len(date)-future_step-past_step-1)):
        if date[i+past_step+future_step]-date[i+past_step+future_step-1] == np.timedelta64(1, timedelta): 
            tmp_x_past = np.stack(x_past[x_past.index.isin(date[i:i+past_step])].values)
            tmp_x_fut = np.stack(x_fut[x_fut.index.isin(date[i+past_step:i+past_step+future_step])].values)            
            yh_tmp =model(torch.from_numpy(tmp_x_past).unsqueeze(0).to(model.device), 
                          torch.from_numpy(tmp_x_fut).unsqueeze(0).to(model.device),
                          adj.to(model.device)).detach().cpu()
            yh.append(F.relu(yh_tmp))
            
            tmp_y = np.vstack(y_group[y_group.index.isin(date[i+past_step:i+past_step+future_step])].values)
            y.append([tmp_y])
        
        else:
             i += past_step+future_step 

    yh = torch.cat(yh)
    y = np.vstack(y) 
    f = y.shape[-1]
    for step in tqdm(range(37, model.future)):
        fig, ax = plt.subplots(nrows = f, 
                                   ncols = 1, 
                                   constrained_layout = True,
                                   figsize = (20,f*3))
    
        for n in range(f):
            ax[n].plot(y[:,step, n], label = 'real')
            ax[n].plot(yh[:,step, n], label = 'estimated')
            ax[n].legend()  

            err = np.mean(np.abs(yh[:,step, n].numpy()-y[:,step, n]))
            ax[n].title.set_text(f"node {n}, step {step} train, err = {err}")
        
        path = os.path.join(config['paths']['fig'], config['setting']['dataset'], 'flows', name)
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, f"step{step+1}.png"))
        plt.close(fig)