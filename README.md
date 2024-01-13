# LAZY-TS for graph diffusion


<!-- TOC --><a name="2-introduction"></a>
##  1. <a name='Introduction'></a>Introduction 
Welcome to this repository! 

This project of mine is centered on exploring the expressivity Graph Neural Networks (GNNs) and their connection with diffusion models on graphs. The goal of this repo is to create a unique library for multi-dimensional (or unidimensional) timeseries regression on static graph. Up to now the model that have been implemented are 
* GLSTM one shot prediction
* GLSTM seq2seq prediction
* GCN_LSTM
* GAT-LSTM one shot prediction
* GAT-LSTM seq2seq

Future release will have
* GCLSTM seq2seq prediction
* ARIMA
* ARMA

Once a certain dataset is given and all the models have been executed, the ranking list is shown.

The order is given by the loss function defined at the begining because is the one that is shared among all the models. This is important to say because there can be models that require additional terms to the loss in order to work, like KL divergence for the VAE model.

Here there is shown a toy example of how the ranking list could be:

|model|score|
|----|----|
|GLSTMseq2seq|2210.542999267578|
|GAT_LSTMseq2seq|2347.313934326172|
|GLSTM|5316.68505859375|
|GAT_LSTM|5443.055419921875|
|GCN_LSTM|5471.723876953125|

Moreover, once everything has been computed, the best model is used for plotting the whole time series with a comparison between real and approximation values:

![image](/Image/timeseries.png)


<!-- TOC --><a name="3-time-series"></a>
##  3. <a name='Timeseries'></a>Time Series 

**Definition [Time Series]**  
A multivariate timeseries is a finite part of a realization from a stochastic process $\{X_t,t\in T\}\subset\mathbb{R}^k$.


As it was said in the definition $X_t\in \mathbb{R}^k$ but it is a result of a concatenation of 2 types of variables:

1. $\mathcal{C} \subset \mathbf{N}^c$ the set of categorical variables.
2. $\mathcal{W} \subset \mathbf{R}^{k-c}$ the set of numerical variables

For this reason it is possible to write $\forall t\in T_0, 
X_t = (c_t|w_t)$ where $c_t\in\mathcal{C}, w_t\in \mathcal{W}$.

***   
**Important**  
For simplicity, it is assumed that the categorical variables are always known, i.e. $\{w_t\}_{t\in T}$ is a predictable process.  
***   
Since this library talks about graph it is important to give a formal definition of graph.

**Definition [Graph]**   
A graph is a pair $G = (V, E)$, where $V$ is the set of the elements of the graph and $E$ is a set of edges, i.e. let $v,u\in V$ then if $v$ is connected with $u$ then $(v,u)\in E$. Moreover, let $u\in V$, the set of neighbours of $u$ is defined as
```math
N(u):=\{v\in V|(u,v)\in E\}
```
Here it is possible to see an example of graph

![image](Image/graph.png)

The graph it is said to be **static** if the adjacency do not vary in time.

<!-- TOC --><a name="4-data-manipulation"></a>
##  4. <a name='Dataset'></a>Dataset
Now it is possible to describes how a dataset is defined and so how is the input of a model.  
As it was said at the beginning $x_t$ is a concatenation of numerical an categorical variables. Thus if $x_t\in \mathbb{R}^k$ then the first $m$ variables are those reffering to numerical variables while the last $k-m$ are referring to the categorical variables.  

Given a dataset `df` the important variables for creating the dataset are:
* `past_step` : number of step to look in the past.
* `future_step`: number of step to look in the future.
* `past_variable`: a list of all the variables that need to be taken from the dataset corresponding to the past
* `future_variable`: a list of all the variables that need to be taken from the dataset corresponding to the past
* `y`: list of numerical variables that need to be predicted
* `adj`: the adjacency matrix of the graph.
* `nodes`: number of nodes of the graph

```
class dataset(Dataset):
    
    def __init__(self, 
                 df: pandas.Dataframe,
                 past_step:int, 
                 future_step:int, 
                 past_variables: list, 
                 y:list,
                 adj: np.array,
                 nodes: int, 
                 timedelta:str = 'D',
                 col_data: str = "data"):
        self.x = []
        self.y = []
        self.adj = adj
        self.past_step = past_step
        self.future_step = future_step
```
Let $p$ be the past steps to be consider and $m$ the future steps. $\forall t\in T$, the input of the models is composed of 
* **the past information**: a 3D-tensor (4D-tensor if it is consider the batch input), where $\forall k\in [0, \cdots, p-1]$
```math
\mathbb{X}_{t-k}^p=\begin{bmatrix}X_{t-k,1}\\ \vdots\\ X_{t-k,|V|}\end{bmatrix}
```
* **the future information** (the predictable process): a 3D-tensor where $\forall k\in [1, \cdots, f]$
```math
\mathbb{X}_{t-k}^p=\begin{bmatrix}X_{t-k,1}\\ \vdots\\X_{t-k,|V|}\end{bmatrix}
```
____
<!-- TOC --><a name="graph-neural-networks-gnns-and-diffusion-models-introduction"></a>
# Graph Neural Networks (GNNs) and Diffusion Models Introduction
<!-- TOC --><a name="5-models"></a>
##  5. <a name='Models'></a>Models

<!-- TOC --><a name="51-gcn"></a>
###  5.1. <a name='GCN'></a>GCN
The Graph Convolutional neural network use the augmented laplacian in order to do the convolution 
Let $G=(V, E)$ be a graph and denote $D, A$ respectively the degree and the adjacency matrix of $G$. Then the augmented Laplacian matrix is 
```math
    \Tilde{L}=\Tilde{D}^{-\frac{1}{2}}L\Tilde{D}^{-\frac{1}{2}}= I-\Tilde{D}^{-\frac{1}{2}}\Tilde{A}\Tilde{D}^{-\frac{1}{2}}
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