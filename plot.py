import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import yaml
import torch.nn.functional as F

def plot(model, 
         config,
         loss_training, 
         loss_validation, 
         name,
         dl_train,
         dl_val, 
         show = False):
    
    fig = px.line({"epochs": range(1,len(loss_training)+1), 
                                   "train": loss_training, 
                                   "validation": loss_validation}, 
                                  x = "epochs", 
                                  y = ["train", "validation"], 
                                  title= f"training loss for {name}")
    fig.add_vline(x = np.argsort(loss_validation)[0]+1)
    fig.add_hline(y = np.min(loss_validation))
    fig.write_html(os.path.join(config['paths']['fig'],'covid', f"loss_gnn_{name}.html"))
    if show:
        fig.show()
    model = model.cpu()
    model.device = torch.device('cpu')
    if config['dataset']['aggregate']:
        batch_train, y_train, adj_train = next(iter(dl_train))
        batch_val, y_val, adj_val = next(iter(dl_val))
        yh_train = model(batch_train.float().to(model.device), adj_train.to(model.device)).detach().numpy()
        yh_val = model(batch_val.float().to(model.device), adj_val.to(model.device)).detach().numpy()
    else:
        yh_train = []
        yh_val = []
        y_train = []
        y_val = []
        for key in dl_train.keys():
            batch_train, yt, adj_train = next(iter(dl_train[key]))
            batch_val, yv, adj_val = next(iter(dl_val[key]))
            yh_train.append(model(batch_train.float().to(model.device), adj_train.to(model.device)).cpu())
            yh_val.append(model(batch_val.float().to(model.device), adj_val.to(model.device)).cpu())
            y_train.append(yt)
            y_val.append(yv)
        
        yh_train = torch.cat(yh_train, -2).transpose(-1,-2)
        yh_val = torch.cat(yh_val, -2).transpose(-1,-2).detach().numpy()
        y_train = torch.cat(y_train, -1).detach().numpy()
        y_val = torch.cat(y_val, -1).detach().numpy()

    lenght = 3*y_val.shape[1]
    fig, ax = plt.subplots(nrows = y_val.shape[1], 
                           ncols = 2, 
                           constrained_layout = True,
                           figsize = (20,lenght))
    
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
    
    path = os.path.join(config['paths']['fig'],'covid', f"{name}.png")
    plt.savefig(path)
    if show:
        plt.show()
    plt.close(fig)

def plot_stream(model, 
                data: pd.DataFrame, 
                config: yaml, 
                time_step: int, 
                adj:torch.tensor,
                name:str, 
                show:bool = False):
    
    past_step = model.past
    fut = np.min((time_step, model.future))
    date = np.sort(data.data.unique())
    y = []
    yh = []
    x_date = []
    for i in range(past_step, len(date)-past_step-fut):
        x = data[data.data.isin(date[i-past_step:i])].drop(columns = 'data')
        y.append(data[data.data == date[i+fut-1]].y.values)
        x_date.append(date[i+past_step-1+fut])
        x = x.values.reshape(1, past_step, 43, config['setting']['in_feat'])
        yh.append(F.relu(model(torch.from_numpy(x).to(model.device), adj[0].to(model.device)).detach()).cpu()[:,fut, :])

    yh = torch.cat(yh,0)
    y = np.vstack(y) 
    f = y.shape[-1]
    fig, ax = plt.subplots(nrows = f, 
                               ncols = 1, 
                               constrained_layout = True,
                               figsize = (20,f*3))
    
    for n in range(f):
        ax[n].plot(y[:,n], label = 'real')
        ax[n].plot(yh[:,n], label = 'estimated')
        ax[n].legend()  
        err = np.mean(np.abs(yh[:,n].numpy()-y[:,n]))
        ax[n].title.set_text(f"node {n}, step {fut} train, err = {err}")
    plt.show()
    
    path = os.path.join(config['paths']['fig'], f"{name}.png")
    plt.savefig(path)
    if show:
        plt.show()
    plt.close(fig)