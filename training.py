import torch
import numpy as np
from tqdm import tqdm
def step(model, 
         dataloader,
         optimizer,
         criterion,
         training: bool = False):

    if training:
        model.train()
    else:
        model.eval()
    
    is_train = True if training else False
    loss_epoch = 0.0
    with torch.set_grad_enabled(is_train):
        for x, y, adj in iter(dataloader):
            x = x.to(model.device).float()
            y = y.to(model.device).float()
            yh = model(x, adj.to(model.device))
            loss = criterion(yh, y)
    
            # Backward and optimize
            if training:
                loss.backward()
                optimizer.step()
                
                # blocking the gradient summation 
                optimizer.zero_grad()
    
            loss_epoch += loss.item()
    loss_epoch/=len(dataloader)

    return loss_epoch


def training(model, 
             train_loader, 
             val_loader, 
             num_epochs, 
             criterion,
             optimizer):
    loss_train = []
    loss_val = []
    be = np.inf
    bm = model
    
    for epoch in tqdm(range(num_epochs)):
            
        # Calculate average loss for the epoch
        loss = step(model = model, 
                         dataloader = train_loader, 
                         optimizer = optimizer, 
                         criterion = criterion,
                         training = True)
        
        loss_train.append(loss)
        
        loss = step(model = model, 
                         dataloader = val_loader, 
                         optimizer = optimizer, 
                         criterion = criterion,
                         training = False)
        
        loss_val.append(loss)
        
        if (loss_val[-1]<be) & (epoch/num_epochs>0.15):
            be = loss_val[-1]
            bm = model
        if (epoch+1)%10 == 0:
            print(f"loss train epoch {epoch+1} == {loss_train[-1]}")
            print(f"loss val epoch {epoch+1} == {loss_val[-1]}")
    return bm, loss_train, loss_val

