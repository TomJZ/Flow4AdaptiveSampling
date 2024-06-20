# Flow modeling and prediction for adaptive sampling
This repository contains the code for training flow models from data. The trained flow model can then be evaluated and used for generating flows.

# How to use
## Data
Training data can be found in /Data/TrainingDataProcessed. Four different types of flows can be found in this folder and is explained below:
### Forced turbulence
These data are similar to chaotic mixing, and are generated from [this notebook](https://github.com/google/jax-cfd/blob/main/notebooks/spectral_forced_turbulence.ipynb)
These data are described by their file names: chaotic_[x]by[x]_[flowtype]_[scaling type].npy denotes a flow simulated on an x by x grid normalized with zzz scaling.
For example, 
* chaotic_40by40_vorticity_minmax_scaled.npy is the vorticity of a 40 by 40 grid chaotic flow, and normalized by a minmax scaling
* chaotic_40by40_grid is the grid information used to simulate chaotic flows on 40 by 40 grid.

### Double Gyre

### Ocean Flows
The ocean flow data is from National Oceanic and Atmospheric Administration [(NOAA)](https://www.noaa.gov/).The data, is on a 30 by 30 grid, with an approximate spatial resolution of 6.6km in both longitude and latitude. The area of ocean considered is within the bounding box between the latitudes 39.360065 and 41.130714. and longitudes -64.240000 and -61.920040

### Vortex Shedding
2D flow around a cylinder. The flow is simulated by solving the Navier-Stoke equations in a 50m by 40m rectangular work space. An inflow stream with uniform velocity profile is imposed on the left boundary, with a zero pressure outflow on the right boundary. The cylinder has a 1m radius and is centered at (0, 0). The Reynolds number of the flow is 200.

## Training
To train a flow model
```python3 main.py```

## Evaluation
## Inference
