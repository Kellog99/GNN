import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

class covid_dataset(Dataset):
    
    def __init__(self, 
                 df:pd,
                 y:pd,
                 adj: np.array,
                 past_step:int, 
                 future_step:int):
        """
        Arguments:
            df (pandas.Dataframe): Path to the csv file with annotations.
            past_step (int): previous step to look back
            future_step (int): future step to look for
        """
        self.x = []
        self.y = []
        self.adj = []
        n = len(df.codice_regione.unique())
        date = df.data.unique()
        date.sort()
        n_provincie = len(df.codice_provincia.unique())
        start = 0
        dt = np.diff(date[:past_step+future_step]) == np.timedelta64(1, 'D')
        while any(not x for x in dt):
            start +=1
            dt = np.diff(date[start:past_step+future_step+ start]) == np.timedelta64(1, 'D')
               
        for i in tqdm(range(start, len(date)-future_step-past_step-1)):
            if date[i+past_step+future_step]-date[i+past_step+future_step-1] == np.timedelta64(1, 'D'): 
                tmp_x = df[df.data.isin(date[i:i+past_step])].drop(columns = "data").values
                tmp_y = y[df.data.isin(date[i+past_step:i+past_step+future_step])].nuovi_casi.values

                self.x.append(tmp_x.reshape(past_step, n_provincie, -1))
                self.y.append(tmp_y.reshape(future_step, -1))
                self.adj.append(np.tile(adj,(past_step,1,1)))
            else:
                i += past_step+future_step
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.adj[idx]
    
