import torch
import torch.nn as nn
import torch.nn.functional as F

class embedding_layer(torch.nn.Module):
    def __init__(self,
                 categorical:list,
                 dim_categorical:int, 
                 concat:bool):
        
        super(embedding_layer, self).__init__()
        self.concat = concat
        self.embedding = nn.ModuleList([nn.Embedding(categorical[i], dim_categorical) for i in range(len(categorical))])

        
    def forward(self, x):
    
        out = []
        # Con l'unsqueeze creo una nuova dimensione ed è come se mi salvassi per ogni features di ogni nodo di ogni batch tutti gli embedding
        # Questo solo se devo sommare tutti gli embedding
        for i in range(len(self.embedding)):
            if self.concat:
                cat_tmp = x[:, :, :, -len(self.embedding)+i].int()
            else:
                cat_tmp = x[:, :,:, -len(self.embedding)+i].int().unsqueeze(-1)
            out.append(self.embedding[i](cat_tmp))

        out = torch.cat(out,-1 if self.concat else -2)
        out = out if self.concat else torch.sum(out, -2)
        out = torch.cat((x[:,:,:,:-len(self.embedding)], out), -1)
        
        return out.float()
        
class pre_processing(torch.nn.Module):
    def __init__(self,
                 in_feat:int, 
                 out_feat:int, 
                 dropout:float):
        
        super(pre_processing, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_features = in_feat, out_features = 128),
                                    nn.ReLU(), 
                                    nn.Dropout(dropout),
                                    nn.Linear(in_features = 128, out_features = 128),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(in_features = 128, out_features = out_feat, bias=False))
        
        
    def forward(self, x):  
        return self.linear(x.float())

class my_gcnn(torch.nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 past:int, 
                 dim_hidden_emb:int = 128):
        super(my_gcnn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.past = past 
        self.lin = nn.Linear(in_channels, 
                             out_channels, 
                             bias = False)
        self.emb = nn.Linear(in_features = in_channels, 
                             out_features = dim_hidden_emb, 
                             bias = False)
        

    def forward(self,
                x0: tuple) -> torch.tensor:
        
        x, A = x0   
        x_emb = self.emb(x)
        pi = torch.einsum('bdjk,bdik->bdij', x_emb, x_emb)

        # Apply the mask to fill values in the input tensor
        # sigmoid(Pi*X*W)
        pi = F.softmax(pi.masked_fill(A == 0., -float('infinity')), -1)
        x = torch.einsum('bpik,bpkj->bpij', pi, x)
        x = F.sigmoid(self.lin(x))
        
        return (x, A)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM weights and biases
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_g = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=-1)

        # Compute the input, forget, output, and cell gates
        i = torch.sigmoid(self.W_i(combined))
        f = torch.sigmoid(self.W_f(combined))
        o = torch.sigmoid(self.W_o(combined))
        g = torch.tanh(self.W_g(combined))

        # Update the cell state and hidden state
        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c
        
        
class GLSTMseq2seq(torch.nn.Module):
    def __init__(self, 
                 in_feat:int, 
                 past: int, 
                 future: int,
                 categorical:list,
                 device, 
                 out_preprocess:int = 128, 
                 dropout: float = 0.1, 
                 dim_categorical:int = 64, 
                 concat:bool = True,
                 num_layer_gnn:int = 1,
                 hidden_gnn:int = 128,
                 hidden_lstm: int = 128, 
                 hidden_propagation:int = 128):
        
        super(GLSTMseq2seq, self).__init__()
        print("GLSTMseq2seq")
        self.in_feat = in_feat         # numero di features di ogni nodo prima del primo GAT        
        self.past = past 
        self.future = future
        self.hidden_gnn = hidden_gnn
        self.hidden_lstm = hidden_lstm
        self.device = device
        ########## PREPROCESSING PART #############        
        # preprossessing dell'input
    
        self.embedding = embedding_layer(categorical = categorical,
                                           dim_categorical = dim_categorical, 
                                           concat = concat)
        self.out_preprocess = out_preprocess
        in_feat_preprocessing = in_feat + (dim_categorical-1)*len(categorical) if concat else in_feat + dim_categorical - len(categorical)
        self.cat_index = in_feat-len(categorical)
        self.pre_processing = pre_processing(in_feat = in_feat_preprocessing, 
                                             out_feat = out_preprocess,
                                             dropout = dropout)
        
        ########## FIRST GNN PART #############
        # LA CGNN mi permette di vedere spazialmente la situazione circostante
        # più layer di CGNN più spazialmente distante arriva l'informazione
        # B x P x N x F
    
        layers = []
        
        for i in range(num_layer_gnn):
            in_channels = self.out_preprocess if i == 0 else hidden_gnn
            layers.append(my_gcnn(in_channels = in_channels, 
                                  out_channels = hidden_gnn, 
                                  past = past))            
        self.gnn = nn.Sequential(*layers)

        self.lstm = LSTMCell(input_size = hidden_gnn, 
                             hidden_size = hidden_lstm)
        
        in_feat_lstm_future = 1 + dim_categorical*len(categorical) if concat else 1 + dim_categorical 
        
        self.lstm_future = LSTMCell(input_size = in_feat_lstm_future, 
                                    hidden_size = hidden_lstm)
        
        self.preprocessing2 =  pre_processing(in_feat = 1+len(categorical), 
                                             out_feat = hidden_lstm, 
                                             dropout = dropout)
        
        self.decoding = nn.Sequential(nn.Linear(in_features = hidden_lstm, 
                                                out_features = hidden_propagation), 
                                      nn.ReLU(), 
                                      nn.Linear(in_features = hidden_propagation,
                                                out_features = hidden_propagation),
                                      nn.ReLU(),
                                      nn.Linear(in_features = hidden_propagation,
                                                out_features = 1))
               
    def forward(self, input, adj):
        ########### pre-processing dei dati ##########
        x = input 
        emb = self.embedding(x)
        x = self.pre_processing(emb)
        nodes = x.shape[-2]
        ########## GNN processing ######################
        x, _ = self.gnn((x, adj))
        
        batch_size, seq_len, nodes, features = x.size()

        # Initialize hidden and cell states
        
        h, c = [torch.zeros(batch_size, nodes, self.hidden_lstm).to(x.device)] * 2
        for t in range(seq_len):
            h, c = self.lstm(x[:, t], (h, c))  
        out = [self.decoding(h).unsqueeze(1)]
        h, c = [torch.zeros(batch_size, nodes, self.hidden_lstm).to(x.device)] * 2
        
        for i in range(self.future-1):
            x_t = torch.cat((out[-1], emb[:,:1,:,self.cat_index:]), -1)
            h, c = self.lstm_future(x_t[:,0], (h,c))
            out.append(self.decoding(h).unsqueeze(1))
        out = torch.cat(out, 1).flatten(-2)
        return  out