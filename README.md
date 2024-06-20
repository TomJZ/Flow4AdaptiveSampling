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
The training loop can be found in ```train.py```
To train a flow model, modify the file ```main.py```. You need to
* Select the training data on the line ```training_data = [data of your choice]```
* Choose your NODE model on the line ```ode_train = NeuralODE([model of your choice])
* Select your training parameters such as the number of epochs, learning rate, etc.
* Specify the path to save your trained model on the line ```save_path = [path_to_save_trained-model]```
* To train only on fully developed flows, you can truncate the data by specifying the index from which the data is used for training. This can be specified with ```train_start_idx``` ,and the length of training data used can be specified by ```train_len```
* To start training, run:

```python3 main.py```

## Models
Model classes are in the file ```models.py```
## Evaluation and Inference
Use evaluation.py for model evaluation and inference.
