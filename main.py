from Utils.Plotters import *
from Utils.Solvers import *
from models import *
from train import *
import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    Load Training Data
    """
    training_data = np.load("Data/Processed/vortex_re200_no_turbulence.npy")
    all_len, nc, x_size, y_size = training_data.shape
    print("All data shape is: ", training_data.shape)

    """
    Training parameters
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ode_solve = Euler
    step_size = 0.01
    loss_arr = []  # initializing loss array
    # initialize NODE model
    ode_train = NeuralODE(VortexConvGaussian().to(device), ode_solve, step_size).double().to(device)
    # ode_train = torch.load("SavedModels/vortex_conv_gaussian_noTurb.pth")['ode_train']
    n_grid = x_size * y_size  # grid size
    vortex_train = training_data
    epochs = 1300
    lookahead = 2
    iter_offset = 0
    lr = 0.001
    save_path = "SavedModels/vortex_conv_gaussian_noTurb"  # file extenssion will be added in training loop
    train_start_idx = 200  # the index from which training data takes from all data
    train_len = 50  # length of training data
    step_skip = 6  # number of steps per time interval
    obs = torch.tensor(vortex_train[train_start_idx:train_start_idx + train_len]).view(train_len, 2, 50,
                                                                                       30).double().to(device)
    obs_t = step_skip * torch.tensor((np.arange(len(obs))).astype(int))
    NOISE_VAR = 0.001  # Variance of gaussian noise added to the observation. Assumed to be 0-mean
    obs[:, :, :, 0:2] = obs[:, :, :, 0:2] + torch.randn_like(obs[:, :, :, 0:2]) * NOISE_VAR
    print("Training data shape is: \n", obs.shape)
    # run the model once
    z_p = ode_train(obs[0].unsqueeze(0), obs_t,
                    return_whole_sequence=True)
    print("z_p size\n", z_p.size())

    # make_color_map(z_p[:, :, :, :].detach().numpy().reshape([train_len, -1]), "Training ConvNN for KS\n",
    #                slice=False, save=None, figure_size=(8, 3))
    # make_color_map(obs[:, :, :, :].detach().numpy().reshape([train_len, -1]), "Training ConvNN for KS\n",
    #                slice=False, save=None, figure_size=(8, 3))
    #
    # plt.show()

    sample_and_grow_PDE(ode_train, obs, obs_t, epochs, lookahead, iter_offset, lr,
                        save_path, plot_freq=20)
