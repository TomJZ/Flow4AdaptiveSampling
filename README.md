* Flow modeling and prediction for adaptive sampling
This repository contains the code for training flow models from data. The trained flow model can then be evaluated and used for generating flows.

*How to use
**Data
Training data can be found in /Data/TrainingDataProcessed. Four different types of flows can be found in this folder and is explained below:
*** Forced turbulence
These data are similar to chaotic mixing, and are generated from [this notebook](https://github.com/google/jax-cfd/blob/main/notebooks/spectral_forced_turbulence.ipynb)
These data are described by their file names: chaotic_[x]by[x]_[flowtype]_[scaling type].npy denotes a flow simulated on an x by x grid normalized with zzz scaling.
For example, 
chaotic_40by40_vorticity_minmax_scaled.npy is the vorticity of a 40 by 40 grid chaotic flow, and normalized by a minmax scaling
chaotic_40by40_grid is the grid information used to simulate chaotic flows on 40 by 40 grid.
 
**Training
**Evaluation
**Inference
