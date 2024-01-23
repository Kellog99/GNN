import torch
import torch.nn as nn
import torch.nn.functional as F

class embedding_layer(torch.nn.Module):
    def __init__(self,
                 categorical:list,
                 dim_categorical:int):
        
        super(embedding_layer, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(categorical[i], dim_categorical) for i in range(len(categorical))])
    def forward(self, x):
    
        out = 0.0
        # Con l'unsqueeze creo una nuova dimensione ed è come se mi salvassi per ogni features di ogni nodo di ogni batch tutti gli embedding
        # Questo solo se devo sommare tutti gli embedding

        for i in range(len(self.embedding)):
            out += self.embedding[i](x[:,:,:, i])    
        return out.float()
        
class pre_processing(torch.nn.Module):
    def __init__(self,
                 in_feat:int, 
                 out_feat:int, 
                 dropout:float):
        
        super(pre_processing, self).__init__()
        self.linear = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(in_features = in_feat, out_features = 128),
                                    nn.ReLU(), 
                                    nn.Linear(in_features = 128, out_features = 128),
                                    nn.ReLU(),
                                    nn.Linear(in_features = 128, out_features = out_feat, bias=False))
        
        
    def forward(self, x):  
        return self.linear(x.float())
    
class my_gcn(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 dim_hidden_emb:int = 128):
        super(my_gcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.emb = nn.Linear(in_features = in_channels, 
                             out_features = dim_hidden_emb, 
                             bias = False)
        

    def forward(self,
                x0: tuple) -> torch.tensor:
        
        x, A = x0   
        x_emb = self.emb(x)        

        # Apply the mask to fill values in the input tensor
        # sigmoid(Pi*X*W)
        D_tilde = torch.diag((torch.sum(A,-1)+1)**(-0.5))
        L_tilde = torch.eye(D_tilde.shape[0]).to(x.device.type)-torch.matmul(torch.matmul(D_tilde, A), D_tilde)
        H = torch.einsum('ik, bskj -> bsij',L_tilde.float(), x_emb)
        x = F.sigmoid(H)
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
        
        
class GCN_LSTM(torch.nn.Module):
    def __init__(self, 
                 in_feat_past:int, 
                 in_feat_fut:int, 
                 past: int, 
                 future: int,
                 categorical_past:list,
                 categorical_future:list,
                 device, 
                 out_preprocess:int = 128, 
                 dropout: float = 0.1, 
                 dim_categorical_past:int = 128, 
                 dim_categorical_future:int = 64, 
                 concat:bool = True,
                 num_layer_gnn_past:int = 1,
                 num_layer_gnn_future:int = 1,
                 out_gnn: int = 128, 
                 hidden_gnn:int = 256,
                 hidden_lstm: int = 128, 
                 hidden_propagation:int = 128):
        
        super(GCN_LSTM, self).__init__()
        self.in_feat_past = in_feat_past         # numero di features di ogni nodo prima del primo GCN        
        self.in_feat_past = in_feat_fut
        self.past = past 
        self.future = future
        self.hidden_gnn = hidden_gnn
        self.hidden_lstm = hidden_lstm
        self.device = device
        self.categorical_past = categorical_past
        self.categorical_fut = categorical_future
        
        ########## PREPROCESSING PART #############        
        self.embedding_past = embedding_layer(categorical = categorical_past,
                                            dim_categorical = dim_categorical_past)
        if len(categorical_future)>0:
            self.embedding_future = embedding_layer(categorical = categorical_future,
                                                    dim_categorical = dim_categorical_future)
        
        in_feat_preprocessing_past = in_feat_past + dim_categorical_past - len(categorical_past)
        self.pre_processing_past = pre_processing(in_feat = in_feat_preprocessing_past, 
                                                 out_feat = out_preprocess, 
                                                 dropout = dropout)
        in_feat_preprocessing_fut = in_feat_fut + dim_categorical_future - len(categorical_future)
        self.pre_processing_fut = pre_processing(in_feat = in_feat_preprocessing_fut, 
                                                 out_feat = out_preprocess, 
                                                 dropout = dropout)

        ########## GNN ############# 
        ##### past
        layers = []
        for i in range(num_layer_gnn_past):
            layers.append(my_gcn(in_channels = out_preprocess if i == 0 else hidden_gnn, 
                                 out_channels = out_gnn if i == num_layer_gnn_past-1 else hidden_gnn))            
        self.gnn_past = nn.Sequential(*layers)

        ##### future
        layers = []
        for i in range(num_layer_gnn_future):
            layers.append(my_gcn(in_channels = out_preprocess if i == 0 else hidden_gnn, 
                                 out_channels = out_gnn if i == num_layer_gnn_future-1 else hidden_gnn))            
        self.gnn_future = nn.Sequential(*layers)

        ######### LSTM ################
        # sia l'embedding del passato che quello del futuro hanno la stessa dimensionalità
        # quindi poso usare lo stesso lstm e prendere come output gli ultimi `fut_step` 
        self.lstm = LSTMCell(input_size = out_gnn, 
                             hidden_size = hidden_lstm)
        self.decoding = nn.Sequential(nn.Linear(in_features = hidden_lstm + future*out_gnn, 
                                                out_features = hidden_propagation), 
                                      nn.ReLU(), 
                                      nn.Linear(in_features = hidden_propagation,
                                                out_features = hidden_propagation),
                                      nn.ReLU(),
                                      nn.Linear(in_features = hidden_propagation,
                                                out_features = future))
               
    def forward(self, x_past, x_fut, adj):        
        ########### pre-processing dei dati ##########
        ##### past 
        emb = self.embedding_past(x_past[:,:,:,-len(self.categorical_past):].int())
        x_past = torch.cat((x_past[:,:,:,:-len(self.categorical_past)], emb), -1)
        x_past = self.pre_processing_past(x_past)
        
        ##### future
        if len(self.categorical_fut)>0:
            emb = self.embedding_future(x_fut[:,:,:,-len(self.categorical_fut):].int())
            x_fut = torch.cat((x_fut[:,:,:,:-len(self.categorical_fut)], emb), -1)    
        x_fut = self.pre_processing_fut(x_fut)
        
        ########## GNN processing ######################
        x_past, _ = self.gnn_past((x_past, adj))
        x_fut, _ = self.gnn_future((x_fut, adj))
        
        ########## LSTM part ###########################
        x_lstm = x_past
        
        batch_size, seq_len, nodes, features = x_lstm.size()
        h, c = [torch.zeros(batch_size, nodes, self.hidden_lstm).to(x_past.device)] * 2
        out = []
        for t in range(seq_len):
            h, c = self.lstm(x_lstm[:, t], (h, c)) 
        x_fut = x_fut.transpose(1,2).flatten(-2)
        tmp = torch.concat((h, x_fut),-1)
        out = self.decoding(tmp)
        return  out