# Graph Neural Networks (GNNs) and Diffusion Models Introduction

Welcome to the repository! This project of mine is centered on exploring the expressivity Graph Neural Networks (GNNs) and their connection with diffusion models on graphs. 
The idea of this repo is to have a unique library for multi multi-dimensional (or even unidimensional, i.e. autoregressive) timeseries regression. Up to now the model that have been implemented are 
* GLSTM one shot prediction
* GLSTM seq2seq prediction
* GAT-LSTM one shot prediction
* GAT-LSTM seq2seq (it is too much expensive)

Future release will have
* GCLSTM one shot prediction with Graph Convolutional network
* GCLSTM seq2seq prediction
* ARIMA
* ARMA

After that all the models have been executed, it is printed the results of the training showing the ranking list of the models.
The position is given by the loss function given in the begining to the model since is the one that is shared among all the models. This is important to say because there can be models that require additional terms to the loss in order to work, like the VAE use the KL divergence.
