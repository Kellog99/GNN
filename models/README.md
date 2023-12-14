# Struttura
Every type of model is characterised by:

* Name model
  * `config.yaml` is the local configuration file which is the one associated with the elements of the model itself
  * `main.py` is the main for each model. It initialize the model and it trains it.
  * `model.py` it contains the definition of the model
  * `training.py` it contains the training procedure for each model. More or less they are similar but it can vary depending on the model