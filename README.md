# Graph Neural Networks (GNNs) and Diffusion Models Introduction

Welcome to the repository! This project of mine is centered on exploring the expressivity Graph Neural Networks (GNNs) and their connection with diffusion models on graphs. 
The idea of this repo is to have a unique library for multi multi-dimensional (or even unidimensional, i.e. autoregressive) timeseries regression. Up to now the model that have been implemented are 
* GLSTM one shot prediction
* GLSTM seq2seq prediction
* GAT-LSTM one shot prediction
* GAT-LSTM seq2seq

Future release will have
* GCLSTM one shot prediction with Graph Convolutional network
* GCLSTM seq2seq prediction
* ARIMA
* ARMA

After that all the models have been executed, it is printed the results of the training showing the ranking list of the models.
The position is given by the loss function given in the begining to the model since is the one that is shared among all the models. This is important to say because there can be models that require additional terms to the loss in order to work, like the VAE use the KL divergence.

### GAT-LSTM
The main point of the GAT's architecture is 
$$
\alpha_{i,j}=
\frac{\exp(\langle a, W h_i||W h_j\rangle)}{\sum_{j\in N(i)}\exp(\langle a, W h_i||W h_j\rangle)} \qquad j\in N(i)
$$
while $\alpha_{i,j}=0$ if $j\notin N(i)$. Since it is very expensive to concatenate two vectors. Thus let $a=(a_1||a_2)$ then
$$
z_{i,j}:=\langle a, v_i||v_j\rangle = \langle a_1, v_i\rangle  +\langle a_2, v_j\rangle  
$$
so let $n=|V|$ then
$$
z_i=\langle v, a_1\rangle \in \R^{n,1} \qquad z_j =\langle v, a_2\rangle\in\R^{n,1}
$$
thus
$$
z = z_1.repeat(1,n)+(z_2.repeat(1,n))^T
$$
This produce the same exponent of the GAT's attention mechanism