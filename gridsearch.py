import os 
import yaml
import time
import pandas as pd
from data import dataset
import pickle
from main import ranking_list

def create_dataset(past_step: int, future_step: int, config: yaml):
    with open(os.path.join(config['paths']['adj'],"adj_totale.pkl"), "rb") as f:
        adj = pickle.load(f)
    data = pd.read_csv(os.path.join(config['paths']['data'], 'covid.csv'), index_col=0)
    data.data = pd.to_datetime(data.data, format="%Y-%m-%d")
    data.rename(columns = {'nuovi_casi':'y'}, inplace=True)

    PATH = os.path.join(config['paths']['data'], config['setting']['dataset'], f"{past_step}_{future_step}.pkl")
    ds = dataset(df = data, 
            past_step = past_step,
            future_step = future_step, 
            nodes = len(data.codice_provincia.unique()),
            future_variables = data.columns[-7:].tolist(), 
            y = ['y'],
            adj = adj, 
            timedelta = 'D')
    with open(PATH, 'wb') as f:
        pickle.dump(ds, f)
if __name__ == "__main__":
    ############### LOAD CONFIG ######################
    with open("./config_env.yaml", 'r') as f:
        config = yaml.safe_load(f)
    steps = list(range(5,51,5))
    for past_step in steps:
        for future_step in steps:
            PATH = os.path.join(config['paths']['data'], config['setting']['dataset'], f"{past_step}_{future_step}.pkl")
            if os.path.exists(PATH)!= True:
                create_dataset(past_step = past_step, 
                               future_step = future_step, 
                               config = config)

    for past_step in steps[2:]:
        for future_step in steps[:-3]:
            txt = f" {config['setting']['dataset']} on {past_step}, {future_step} "
            x = txt.center(80, "#")
            print(x) 

            

            config['setting']['past_step'] = past_step
            config['setting']['future_step'] = future_step
            ranking_list(config=config)
            time.sleep(10)
    txt = f" testing on {config['setting']['dataset']} "
    x = txt.center(80, "#")
