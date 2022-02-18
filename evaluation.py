import tqdm
import numpy as np
from torch import Tensor
import torch
import matplotlib.pyplot as plt

mode = "full"  # whether the model is the simplied planar model

data_path = "Data/uncertain_params/traj_data_mpc_aerodrag_radius3.npy"
model_path = "SavedModels/full_train_on_2traj.pth"

torch_model = torch.load(model_path, map_location=torch.device('cpu'))
ode_train = torch_model['ode_train']
device = torch.device('cpu')
ode_train.func.device = device
train_loss_arr = torch_model['train_loss_arr']
print("Final loss is: \n", train_loss_arr[-1])

if mode!="planar": 
    val_loss_arr = torch_model['val_loss_arr']


if mode=="rigid":
    with open(data_path, 'rb') as f:
        train_set = np.load(f)
    set_len = train_set.shape[0]
    train_set = train_set.reshape([set_len, -1])
    train_set = train_set[:, :]
    data_length = set_len  # the length of data to keep

    titles = ["x", "y", "z", "x_dot", "y_dot", "z_dot", "x_ddot", "y_ddot", "z_ddot"]

    def plot_all(train_set, new_set=None):
        fig = plt.figure(figsize=(15, 15))
        for i in range(9):
            ax = fig.add_subplot(3, 3, i + 1)
            ax.plot(train_set[:, i])
            if new_set is not None:
                ax.plot(new_set[:, i])
            ax.set_title(titles[i])
        plt.savefig("SavedModels/hybrid_state.png", format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)

    plot_len = set_len - 1
    
    ode_input = Tensor(train_set[0]).view(1, 9).to(device)
    pred_traj = [ode_input]
    print("Generating Predictions")
    for i in tqdm.tqdm(range(plot_len)):
        z1 = ode_train(ode_input, Tensor(np.arange(2))).squeeze(1)
        ode_input = torch.cat([z1[:, :6], Tensor(train_set[i + 1, 6:]).unsqueeze(0).to(device)], 1)
        pred_traj.append(ode_input)

    pred_traj = torch.cat(pred_traj).cpu().detach().numpy()
    # plotting the trajectories
    plot_all(train_set[:plot_len], pred_traj)
    # plotting the loss array

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(train_loss_arr[100:], label='Training loss')
    ax.plot(val_loss_arr[100:], label='Validation loss')
    ax.legend()
    plt.savefig("SavedModels/hybrid_loss.png", format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)

elif mode=="planar":
    with open(data_path, 'rb') as f:
        train_set = np.load(f)
    set_len = train_set.shape[0]
    train_set = train_set.reshape([set_len, -1])
    train_set = train_set[:, :]
    data_length = set_len  # the length of data to keep

    titles = ["y", "z", "roll angle", "yd", "zd", "roll rate", "thrust", "moment"]

    def plot_all(train_set, new_set=None):
        fig = plt.figure(figsize=(25, 5))
        for i in range(8):
            ax = fig.add_subplot(2, 4, i + 1)
            ax.plot(train_set[:, i])
            if new_set is not None:
                ax.plot(new_set[:, i])
            ax.set_title(titles[i])
        plt.savefig("SavedModels/simple_aerodrag_state.png", format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)


    plot_len = set_len - 1
    
    ode_input = Tensor(train_set[0]).view(1, 8).to(device)
    pred_traj = [ode_input]
    for i in range(plot_len):
        z1 = ode_train(ode_input, Tensor(np.arange(2))).squeeze(1)
        ode_input = torch.cat([z1[:, :6], Tensor(train_set[i + 1, 6:]).unsqueeze(0).to(device)], 1)
        pred_traj.append(ode_input)

    pred_traj = torch.cat(pred_traj).cpu().detach().numpy()
    # plotting the trajectories
    plot_all(train_set[:plot_len], pred_traj)
    # plotting the loss array

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(train_loss_arr[100:], label='Training loss')
    ax.plot(val_loss_arr[100:], label='Validation loss')
    ax.legend()
    plt.savefig("SavedModels/simple_aerodrag_loss.png", format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()
elif mode=="full":
    with open(data_path, 'rb') as f:
        train_set = np.load(f)
    set_len = train_set.shape[0]
    train_set = train_set.reshape([set_len, -1])
    train_set = train_set[::1, 1:18]
    set_len = train_set.shape[0]
    data_length = set_len # the length of data to keep

    def plot_all(train_set, new_set=None):
        titles = ["pos_x", "pos_y", "pos_z",
                  "v_x", "v_y", "v_z",
                  "q1", "q2", "q3", "q4",
                  "w_x", "w_y", "w_z", "thrust",
                  "mom_x", "mom_y", "mom_z"]

        fig = plt.figure(figsize=(25,5))
        for i in range(17):
            ax = fig.add_subplot(2,9,i+1)
            ax.plot(train_set[:,i])
            if new_set is not None:
                ax.plot(new_set[:, i])
            ax.set_title(titles[i])
        
        plt.savefig("SavedModels/3layer_aerodrag_state.png", format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    
    plot_len = set_len-1
    
    ode_input = Tensor(train_set[0]).view(1, 17).to(device)
    pred_traj = [ode_input]
    print("Generating Trajectories")
    for i in tqdm.tqdm(range(plot_len)):
        z1 = ode_train(ode_input, Tensor(np.arange(2))).squeeze(1)
        ode_input = torch.cat([z1[:,:13], Tensor(train_set[i+1, 13:]).unsqueeze(0).to(device)],1)
        pred_traj.append(ode_input)

    pred_traj = torch.cat(pred_traj).cpu().detach().numpy()
    # plotting the trajectories
    plot_all(train_set[:plot_len], pred_traj)
    # plotting the loss array

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(train_loss_arr[100:], label='Training loss')
    ax.plot(val_loss_arr[100:], label='Validation loss')
    ax.legend()
    plt.savefig("SavedModels/3layer_aerodrag_loss.png", format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()

