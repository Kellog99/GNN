import os 
import yaml
import time
from data import dataset
import pickle
from main import ranking_list

if __name__ == "__main__":
    ############### LOAD CONFIG ######################
    with open("./config_env.yaml", 'r') as f:
        config = yaml.safe_load(f)
    steps = list(range(5,51,5))
    for past_step in steps:
        for future_step in steps:
            txt = f" {config['setting']['dataset']} on {past_step}, {future_step} "
            x = txt.center(80, "#")
            print(x) 
            config['setting']['past_step'] = past_step
            config['setting']['future_step'] = future_step
            ranking_list(config=config)
            time.sleep(10)
    txt = f" testing on {config['setting']['dataset']} "
    x = txt.center(80, "#")