# Graph Neural Networks (GNNs) and Diffusion Models Introduction

Welcome to the repository! This project of mine is centered on exploring the expressivity Graph Neural Networks (GNNs) and their connection with diffusion models on graphs. 
The idea of this repo is to have a unique library for multi multi-dimensional (or even unidimensional, i.e. autoregressive) timeseries regression. Up to now the model that have been implemented are 
* GLSTM one shot prediction
* GLSTM seq2seq prediction

Future release will have
* GCLSTM one shot prediction
* GCLSTM seq2seq prediction
* ARIMA
* ARMA

After that all the models have been executed, it is printed the results of the training showing the ranking list of the models.
The position is given by the loss function given in the begining to the model since is the one that is shared among all the models. This is important to say because there can be models that require additional terms to the loss in order to work, like the VAE use the KL divergence.

## Why GNNs?
### 1. Graph Representation

Graphs are a versatile way to represent complex relationships and dependencies in data. Many real-world systems, such as social networks, biological networks, and citation networks, can be effectively modeled as graphs. GNNs excel in capturing and leveraging the intricate structure of these graphs for various tasks.

### 2. Learning Graph-Structured Data

Traditional neural networks are not well-suited for handling graph-structured data. GNNs, on the other hand, are designed to operate on graph-structured data and can effectively capture local and global dependencies within the graph.

### 3. Node and Graph-Level Representations

GNNs can generate meaningful embeddings for nodes and entire graphs, allowing for efficient representation learning. This is particularly valuable for tasks like node classification, link prediction, and graph classification.

### 4. Message Passing Mechanism

GNNs use a message passing mechanism, where nodes exchange information with their neighbors iteratively. This enables the model to incorporate information from the entire graph, making it powerful for tasks that involve global context.

## Connection with Diffusion Models
### 1. Information Flow

Diffusion models describe how information or influence spreads through a network over time. GNNs can be seen as a neural network instantiation of diffusion models, where information is propagated through the graph via message passing.

### 2. Learning Dynamics

GNNs inherently capture the dynamics of information propagation on graphs. By learning the parameters of the model, GNNs effectively capture how information diffuses through the graph structure, making them suitable for tasks related to influence prediction, epidemic modeling, and more.

### 3. Diffusion-Based Tasks

Tasks such as predicting the spread of innovations, identifying influential nodes, or understanding the dynamics of information cascades can be approached using both GNNs and diffusion models. GNNs provide a data-driven way to learn the diffusion process directly from observed data.
