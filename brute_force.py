import os
import numpy as np
import yaml
import importlib.util as imp
from functools import partial
import argparse
from main import linfty, get_dataloader
import pandas as pd
from data import dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get main variables')
    parser.add_argument('--config', type=str,
                    help='path for the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    name_models = os.listdir(config['paths']['list_models'])
    models = [f"{i}- {mod}\n" for i, mod in enumerate(name_models)]
    print("".join(models))
    print("Choose a model")
    # input
    input1 = int(input())
    
    loss_function = partial(linfty, alpha = float(config['setting']['alpha']))
    df_train, df_val = get_dataloader(config = config)
    scores = {}

    print("the model that has been chosen is ", models[input1][3:-1])
    model = models[input1][3:-1]
    for d in np.linspace(start = 0.0, stop = 0.9, num = 10):
        config['setting']['dropout'] = d
        file_path = os.path.join(config['paths']['list_models'], model, 'main.py')
        print(file_path)
        # Create a module spec
        spec = imp.spec_from_file_location('main.py', file_path)
        # Import the module
        module = imp.module_from_spec(spec)
        spec.loader.exec_module(module)
        be = module.get_model(df_train = df_train, 
                        df_val = df_val, 
                        config_env = config,
                        loss_function = loss_function)
        scores[f"{d}"] = be
    scores = pd.DataFrame(scores.items(),columns=['name','score'])
    print(scores)
    scores.to_csv(f"./{model}_dropout.csv")