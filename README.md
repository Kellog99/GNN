# LAZY-TS for graph diffusion
<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->
- [LAZY-TS for graph diffusion](#lazy-ts-for-graph-diffusion)
  - [1. Introduction](#1-introduction)
  - [3. ](#3-)
  - [4. Data Manipulation](#4-data-manipulation)
- [Graph Neural Networks (GNNs) and Diffusion Models Introduction](#graph-neural-networks-gnns-and-diffusion-models-introduction)
  - [5. Models](#5-models)
    - [5.1. GCN](#51-gcn)
    - [5.2. GAT-LSTM](#52-gat-lstm)
- [TO-DO](#to-do)

<!-- TOC end -->


<!-- TOC --><a name="2-introduction"></a>
##  1. <a name='Introduction'></a>Introduction 
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

The order is given by the loss function given in the begining to the model since is the one that is shared among all the models. This is important to say because there can be models that require additional terms to the loss in order to work, like the VAE use the KL divergence.

Here there are the results of this procedure for the telephone dataset:

|model|score|
|----|----|
|GLSTMseq2seq|2210.542999267578|
|GAT_LSTMseq2seq|2347.313934326172|
|GLSTM|5316.68505859375|
|GAT_LSTM|5443.055419921875|
|GCN_LSTM|5471.723876953125|

Once everything has been computed, with the best model it is plotted the whole time series

<!-- TOC --><a name="3-time-series"></a>
##  3. <a name='Timeseries'></a>

**Definition**   
A multivariate timeseries is a finite part of a realization from a stochastic process $\{X_t,t\in T\}$ in $\mathbb{R}^k$, i.e. $\{X_t,t\in T_0\}\subseteq\mathbb{R}^k$ with $T_0\subset T$


In this case $\mathbb{R}^k$ is the result of a concatenation of 2 space:

1. $\mathcal{C} \subset \mathbf{N}^c$ the set of categorical variables.
2. $\mathcal{W} \subset \mathbf{R}^{k-c}$ the set of numerical variables

For this reason it is possible to write $\forall t\in T_0, 
X_t=c_tw_t$ with $c_t\in\mathcal{C}, w_t\in \mathcal{W}$
where $c_tw_t$ means concatenation.

***   
**Important**  
We assume that the categorical variables are always known, i.e. $\mathcal{C}$ is known $\forall t$.  

***   

Let $t\in T$ then it is possible to define the following spaces: 

  * $\mathcal{F}\subset\mathbf{R}^{f}$: the set of all the variables that are always available.
  * $\mathcal{P}\subset\mathbf{R}^{p}$: the set of variables known up to time $t$,  
* $\mathcal{T}\subset\mathcal{P}\subset\mathbf{R}^{s}$: the set of target variables. 

**Observation**  
$\mathcal{F},\mathcal{P}$ are the result of a concatenation with $\mathcal{C}$ since it is always known.
Moreover $\mathcal{P}$ contains the variables of $\mathcal{T}$.

<!-- TOC --><a name="4-data-manipulation"></a>
##  4. <a name='DataManipulation'></a>Data Manipulation
Unlike a classical time series, this type of dataset contains for each timestep as many multidimensional vectors as there are nodes in the graph. The features of each node for each step in the past are equal.

At the moment I assume that the variables associated with the nodes regarding the future are only categorical: date and node name.

the variables of a node are of two types 
1. numeric
2. categorical 

The numeric variables include the study target variable `y` while the last k columns of the dataset are associated with the k categorical variables associated with the nodes

____
|y|number|date|hour|id_workstation | number|
|----|----|----|----|----|----|
11.2|23|2023-09-09 |18|1027|50
____
<!-- TOC --><a name="graph-neural-networks-gnns-and-diffusion-models-introduction"></a>
# Graph Neural Networks (GNNs) and Diffusion Models Introduction
<!-- TOC --><a name="5-models"></a>
##  5. <a name='Models'></a>Models

<!-- TOC --><a name="51-gcn"></a>
###  5.1. <a name='GCN'></a>GCN
The Graph Convolutional neural network use the augmented laplacian in order to do the convolution 
```math
Let $G=(V, E)$ be a graph and denote $D, A$ respectively the degree and the adjacency matrix of $G$. Then the augmented Laplacian matrix is 
    \[
    \Tilde{L}=\Tilde{D}^{-\frac{1}{2}}L\Tilde{D}^{-\frac{1}{2}}= I-\Tilde{D}^{-\frac{1}{2}}\Tilde{A}\Tilde{D}^{-\frac{1}{2}}
    \]
```
<!-- TOC --><a name="52-gat-lstm"></a>
###  5.2. <a name='GAT-LSTM'></a>GAT-LSTM
This architecture is based on the concept of GAT (Graph attention network) for embedding the features of the nodes and then using an LSTM for generating the stream of data.

The concept of GAT architecture is taken from this [paper](https://arxiv.org/abs/1710.10903).

Let $G=(V,E)$ be a graph and $i\in V$. The set of neighbour of $i$ is defined by $N(i)$.  This is important since the main point of the GAT's architecture is 
```math
\alpha_{i,j}=\left\{\begin{aligned}
&\frac{\exp(\langle a, W h_i||W h_j\rangle)}{\sum_{j\in N(i)}\exp(\langle a, W h_i||W h_j\rangle)}&& j\in N(i)\\
&\alpha_{i,j}=0 && j\notin N(i)
\end{aligned}\right.
```
Since it is very expensive to concatenate two vectors. Thus let $a=(a_1||a_2)$ then
```math
z_{i,j}:=\langle a, v_i||v_j\rangle = \langle a_1, v_i\rangle  +\langle a_2, v_j\rangle  
```
so if $n=|V|$ then
```math
z_i=\langle v, a_1\rangle \in \mathbb{R}^{n,1} \qquad z_j =\langle v, a_2\rangle\in\mathbb{R}^{n,1}
```
thus
```math
z = z_1.repeat(1,n)+(z_2.repeat(1,n))^T
```
This produce the same exponent of the GAT's attention mechanism which is then normalized with a `softmax` operation.

<!-- TOC --><a name="to-do"></a>
# TO-DO
1. The global configuration file should contains all the paths to the local configuration file 
   1. Should it be a whole folder by itself that contains all the files ore one in each directory?
2. `PytorchLightning`
3. Create one whole global training procedure 
   1. How to inser the scheduler into the training procedure.
4. `DataManipulation.py` : it contains the class for organizing the dataset:
    * The input should be the dataset, `past_step`, `future_step`.
    * The last columns are for categorical variables
    * The input of the dataset for the categorical variables is a list that contain the corresponding range of the variables in the dataset.
5. Creare una classe Trainer in modo tale che posso richiamarla per fare evaluation con metriche diverse
6. la funzione plot in `plot.py` deve mostrare l'ndamento ordinato nel tempo non la semplice rappresentazione con la distanza