import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml
import os
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





def fill_dataset(df: pd, 
                 config:yaml):
    # questa procedura serve solo ad inserire i dati sulle province mancanti
    d = []
    feat = df.shape[1]
    codici = list(df.codice_provincia.unique())
    codice_reg_prov = {tuple(x) for x in df[config['dataset']['col_categorical_prov']].values.tolist()}
    codice_reg_prov = pd.DataFrame(codice_reg_prov, columns=config['dataset']['col_categorical_prov'])
    j = 0
    for data in tqdm(df.data.unique(), desc = "filling"):
        tmp = df[df.data == data]
        if len(tmp) == len(codici):
            d.append(tmp)
        else:
            j+=1
            # individue le province mancanti
            codici_mancanti = [x for x in codici if x not in list(tmp.codice_provincia.unique())]
            code_regioni_mancanti = [codice_reg_prov[codice_reg_prov.codice_provincia == x].codice_regione.values[0] for x in codici_mancanti]
            tmp_mancante = pd.DataFrame(np.zeros((len(codici_mancanti),feat)), columns = df.columns)
            
            tmp_mancante.data = data
            tmp_mancante.codice_provincia = codici_mancanti
            tmp_mancante.codice_regione = code_regioni_mancanti
            tmp = pd.concat((tmp, tmp_mancante))
            if len(tmp) != len(codice_reg_prov):
                import pdb
                pdb.set_trace()
            d.append(tmp)
    df = pd.concat(d)

    bilancio = {}
    for csv in os.listdir(config['data']['raw_data']):
        if 'Bilancio demografico' in csv:
            tmp = pd.read_csv(f"./data/{csv}")
            tmp = tmp[tmp.Sesso == "Totale"][['Codice provincia', 'Popolazione al 1° gennaio']]
            tmp.rename(columns={'Codice provincia': 'codice_provincia', 
                                'Popolazione al 1° gennaio': 'popolazione'}, inplace=True)
            anno = int(csv.split(" ")[-1][:4])
            bilancio[anno] = tmp
    merge = []
    for anno in bilancio.keys():
        if anno != 2022:
            tmp = df[(df.data>pd.to_datetime(anno, format = '%Y'))&(df.data<pd.to_datetime(anno+1, format = '%Y'))]
        else:
            tmp = df[df.data>pd.to_datetime(anno, format = '%Y')]
        bi = bilancio[anno]
        bi = bi[bi.codice_provincia.isin(codice_reg_prov.codice_provincia.unique().tolist())]
        tmp = pd.merge(tmp, bi, on = 'codice_provincia', how ='left')
        merge.append(tmp)
    
    df = pd.concat(merge)    
    # Qua ordino le colonne come voglio altrimenti avrei variabile numeriche dove sono quelle categoriche
    df = df[config['dataset']['col_data'] + config['dataset']['col_numerical_prov']+ ['popolazione'] + config['dataset']['col_numerical_reg']+config['dataset']['col_categorical_prov']]
    df = df.drop(columns=["denominazione_regione", "denominazione_provincia"])
    df = df.sort_values(by = config['dataset']['ordering']).drop_duplicates(config['dataset']['ordering']).reset_index(drop = True)
    
    y = df[['data','codice_provincia', 'codice_regione', 'nuovi_casi']]    
    
    ## normalization for each region
    d = []
    col = config['dataset']['numerical']
    remaining = [x for x in list(df.columns) if (x not in col)&(x!='data')]
    for cod in df.codice_regione.unique():
        tmp = df[df.codice_regione==cod]        
        index = int(len(tmp)*0.8)
        for column in col:
            min = tmp[column].values[:index].min()
            max = tmp[column].values[:index].max()
            tmp[column] = (tmp[column]-min)/(max-min)            
        
        d.append(tmp)
        
    tmp = pd.concat(d,0)
    df = tmp.sort_values(by = config['dataset']['ordering']).reset_index(drop = True)  
    return df, y
    
def get_dataset(config: yaml):
    files = ["covid","target", "codice_reg_prov", "covid_province", "covid_regioni"]
    agg = "aggregate" if config['dataset']['aggregate'] else "not_aggregate"
    exists = [os.path.exists(os.path.join(config['paths']['data'], f"{file}.csv")) for file in files]
    
    if  np.all(exists):
        out = ()
        for file in files:
            tmp = pd.read_csv(os.path.join(config['paths']['data'], 'aggregate', f"{file}.csv"), index_col=0)
            if "data" in list(tmp.columns):
                tmp['data'] = pd.to_datetime(tmp['data'], format='%Y-%m-%d')
            out += (tmp,)
        return out[0], out[1]
        
    else:
        ########## I create take the dataset for the provinces
        if os.path.exists(os.path.join(config['paths']['raw_data'],"covid_province.csv")):
            provincials = pd.read_csv(os.path.join(config['paths']['raw_data'],"covid_province.csv"))
            provincials['data'] = pd.to_datetime(provincials['data'], format='%Y-%m-%d')
        else:
            data = []
            for csv in tqdm(os.listdir(os.path.join(config['paths']['raw_data'],"dati-province")), desc = "provincia"):
                if csv.split(".")[-1] == "csv":
                    data.append(pd.read_csv(os.listdir(os.path.join(config['paths']['raw_data'],"dati-province",csv))))
            
            
            data = pd.concat(data)
            data.reset_index(drop = True, inplace=True)
            data.data = data.data.apply(lambda x: x.split("T")[0])
            data.data = pd.to_datetime(data['data'], format='%Y-%m-%d')
            
            # Riduco il dataset con le variabili che servono
            # Inoltre trasformo la conta totale dei positivi per ogni regione in nuovi positivi
            data.rename(columns={'totale_casi': 'nuovi_casi'}, inplace=True)
            provincials = data[config['dataset']['col_data'] + config['dataset']['col_categorical_prov'] + config['dataset']['col_numerical_prov']]
            provincials = provincials.sort_values(by = config['dataset']['ordering'])
            provincials = provincials.drop_duplicates()
            provincials = provincials[-(provincials.codice_provincia>200)]
            provincials = provincials[-provincials.codice_regione.isin([21,22])].reset_index(drop = True)
            provincials.denominazione_regione = provincials.denominazione_regione.replace(['P.A. Bolzano', 'P.A. Trento'], "Trentino")
            
            tmp = provincials[['data','codice_provincia', 'nuovi_casi']]
            tmp = tmp.groupby('codice_provincia').diff().dropna().drop(columns = "data")
            # Non ha senso avere questi valori
            tmp = tmp[-tmp.nuovi_casi<0]
            
            #faccio un merge sugli indici
            provincials = pd.merge(provincials.drop(columns = "nuovi_casi"),tmp, left_index=True, right_index=True)
            provincials = provincials.reset_index(drop=True)
            provincials.to_csv(os.path.join(config['paths']['data'],"covid_province.csv"))

                    
        if os.path.exists(os.path.join(config['paths']['data'], "covid_regioni.csv")):
            regions = pd.read_csv(os.path.join(config['paths']['data'],"covid_regioni.csv"))
            regions['data'] = pd.to_datetime(regions['data'], format='%Y-%m-%d')
        else:
            ########## I create take the dataset for the regions
            data = []
            for csv in tqdm(os.listdir(config['paths']['dati-regioni']), desc = "regione"):
                if csv.split(".")[-1] == "csv":
                    data.append(pd.read_csv(os.path.join(config['paths']['dati-regioni'],csv)))
            data = pd.concat(data).reset_index(drop = True)
            
            # Trento
            data.codice_regione = data.codice_regione.replace(21, 4)
            # Bolzano
            data.codice_regione = data.codice_regione.replace(22, 4)
            data.denominazione_regione = data.denominazione_regione.replace(['P.A. Bolzano', 'P.A. Trento'], "Trentino")
            
            data.data = data.data.apply(lambda x: x.split("T")[0])
            data.data = pd.to_datetime(data['data'], format='%Y-%m-%d')
            regions = data[config['dataset']['col_data'] + config['dataset']['col_categorical_reg'] + config['dataset']['col_numerical_reg']]
            regions = regions.drop_duplicates().reset_index(drop = True)
            regions.to_csv(os.path.join(config['paths']['data'],"covid_regioni.csv"))
                    
        # Creo il dataset completo 
        if os.path.exists(os.path.join(config['paths']['data'],"covid.csv")):
            df = pd.read_csv(os.path.join(config['paths']['data'], "covid.csv"), index_col=0)
            df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
            y = pd.read_csv(os.path.join(config['paths']['data'], "target.csv"), index_col=0)
            y['data'] = pd.to_datetime(y['data'], format='%Y-%m-%d')
        else:
            df = pd.merge(provincials, regions, 
                          how="left",  
                          on = config['dataset']['merge']) 
            df = df.sort_values(by = config['dataset']['ordering']).drop_duplicates(['data','codice_provincia']).reset_index(drop = True)
            df = df[config['dataset']['col_data'] + config['dataset']['col_numerical_prov']+config['dataset']['col_numerical_reg']+config['dataset']['col_categorical_prov']]

            codice_reg_prov = {tuple(x) for x in df[config['dataset']['col_categorical_prov']].values.tolist()}
            codice_reg_prov = pd.DataFrame(codice_reg_prov, columns=config['dataset']['col_categorical_prov'])
            df, y = fill_dataset(df, config)
            y.to_csv(os.path.join(config['paths']['data'],f"target.csv"))
            if ~config['dataset']['aggregate']:
                for codice in df.codice_regione.unique():
                    tmp = df[df.codice_regione == codice].reset_index(drop = True)
                    tmp.to_csv(os.path.join(config['paths']['data'],f"not_aggregate/covid_{codice}.csv"))
                    y.to_csv(os.path.join(config['paths']['data'],f"not_aggregate/target_{codice}.csv"))
            codice_reg_prov.to_csv(os.path.join(config['paths']['data'],"codice_reg_prov.csv"))
            df.to_csv(os.path.join(config['paths']['data'],"covid.csv"))
            
    return df, y

