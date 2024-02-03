import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

class Trainer():
    def __init__(self, 
                 model, 
                 PATH:str,
                 loss_function: callable, 
                 optimizer, 
                 gamma_scheduler: float = 0.9,
                 step: int = 300, 
                 reset_loss:bool = True):
        """
        model : is the model to be optimized
        PATH : is the path where the model needs to be saved
        loss_function : is the loss function that will be used during the training procedure
        optimizer : is the strategy that it is needed 
        reset_loss : if I Execute multiple training with the same model 
        gamma_scheduler (float) : 1.0 by default is the coefficient of the scheduler
        step (int) : it is the frequency of the step
        """

        self.model = model 
        self.PATH = PATH
        self._compute_loss = loss_function
        self.optimizer = optimizer
        self.scheduler = ExponentialLR(optimizer= optimizer, gamma = gamma_scheduler)
        self.step_scheduler = step
        self.reset_loss = reset_loss

        self.loss_train = []
        self.loss_val = []
        self.reset_loss = bool

    def fit(self, 
            train_loader:DataLoader, 
            val_loader:DataLoader, 
            epochs:int = 1):

            if self.reset_loss:
                self.loss_train = []
                self.loss_val = []
            be = np.inf
            bm = self.model

            for epoch in tqdm(range(epochs)):
                
                # train
                self._train(train_loader)
                
                # validate
                self._validate(val_loader)

                
                if (epoch+1)%self.step_scheduler ==0:
                    self.scheduler.step()
                    
                if (self.loss_val[-1]<be) & (epoch/epochs>0.15):
                    be = self.loss_val[-1]
                    bm = self.model
                if (epoch+1)%1 == 0:
                    sys.stdout.flush()
                    print(f"loss train epoch {epoch+1} == {self.loss_train[-1]}")
                    print(f"loss val epoch {epoch+1} == {self.loss_val[-1]}")
            torch.save(bm.state_dict(), self.PATH)
            self.model = bm
            
    def _train(self, loader):
        
        # put model in train mode
        self.model.train()
        loss_epoch  = 0
        for x_past, x_fut, y, adj in iter(loader):
            yh = self.model(x_past.to(self.model.device).float(), x_fut.to(self.model.device).float(), adj[0].to(self.model.device).float())
            loss = self._compute_loss(yh, y.to(self.model.device).float())
    
            # Backward and optimize

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_epoch += loss.item()
        
        self.loss_train.append(loss_epoch/len(loader))

    def _validate(self, loader):
        # put model in evaluation mode
        self.model.eval()
        loss_epoch  = 0
        with torch.no_grad():
           for x_past, x_fut, y, adj in iter(loader):
                yh = self.model(x_past.to(self.model.device).float(), x_fut.to(self.model.device).float(), adj[0].to(self.model.device).float())
                loss = self._compute_loss(yh, y.to(self.model.device).float())
                loss_epoch += loss.item()
        self.loss_val.append(loss_epoch/len(loader))