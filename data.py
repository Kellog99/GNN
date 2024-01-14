import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

class dataset(Dataset):
    
    def __init__(self, 
                 df:pd,
                 adj: np.array,
                 nodes: int, 
                 past_step:int, 
                 future_step:int, 
                 timedelta:str,
                 categorical_variables: list, 
                 col_data: str = "data"):
        """
        Arguments:
            past_step: number of step to look in the past.
            future_step: number of step to look in the future.
            past_variable: a list of all the variables that need to be taken from the dataset corresponding to the past
            future_variable: a list of all the variables that need to be taken from the dataset corresponding to the past
            y: list of numerical variables that need to be predicted
            adj: the adjacency matrix of the graph.
            nodes: number of nodes of the graph
        """

        self.x = []
        self.x_fut = []
        self.y = []
        self.adj = adj
        date = df[col_data].unique()
        date.sort()
        start = 0
        dt = np.diff(date[:past_step+future_step]) == np.timedelta64(1, timedelta)
        while any(not x for x in dt):
            start +=1
            dt = np.diff(date[start:past_step+future_step+ start]) == np.timedelta64(1, timedelta)
        
        for i in tqdm(range(start, len(date)-future_step-past_step-1)):
            if date[i+past_step+future_step]-date[i+past_step+future_step-1] == np.timedelta64(1, timedelta): 
                tmp_x = df[df[col_data].isin(date[i:i+past_step])].drop(columns = col_data).values
                tmp_y = df[df[col_data].isin(date[i+past_step:i+past_step+future_step])]
                
                self.x_fut.append(tmp_y[categorical_variables].values.reshape(future_step, nodes, -1))
                self.x.append(tmp_x.reshape(past_step, nodes, -1))
                self.y.append(tmp_y.y.values.reshape(future_step, -1))
            else:
                i += past_step+future_step
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.x_fut[idx], self.y[idx], self.adj