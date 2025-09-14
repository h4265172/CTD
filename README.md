# CTD

This GitHub repository contains code for Clustered Temporal Decomposition. 
## Jupyter Notebooks
* ```Simple_Demo_with_Instructions.ipynb``` demonstrates appropriate usage of the above files through a simulation experiment.
* ```Simulation_Analysis_CTD-GM-AR.ipynb``` and ```Simulation_Analysis_CTD-GM-TCN.ipynb``` replicate the simulation experiments included in the manuscript.

## Code for architecture
* See ```CTD_ARIMA.py``` and ```CTD_TCN.py``` for the two main Clustered Temporal Decomposition training classes.
* ```data_genarator.py``` contains the data-generating function for simulation experiments.
* ```util.py``` contains the performance measure functions.
* ```data_loader.py``` contains a mini-batch loading function for stochastic gradient descent.



Supplementary material (proofs, additional graphical illustrations and tables) can be found in https://drive.google.com/file/d/1AQ5mMlZPOKv65ROTPBVKRDuJ8LkHamql/view?usp=sharing.
