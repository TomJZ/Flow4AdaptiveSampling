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
    downsize_ratio = 2
    dg_data = np.load("Data/TrainingDataProcessed/dg_flow_field.npy")[:, :, ::downsize_ratio, ::downsize_ratio][:, :, :50, :50]
    vortex_square_data = np.load("Data/TrainingDataProcessed/vortex_re200_with_turbulence.npy")[:, :, :30, :30]
    noaa_data = np.load("Data/TrainingDataProcessed/noaa_flow_field.npy")
    chaotic_data_80by80 = np.load("Data/TrainingDataProcessed/chaotic_80by80.npy").reshape([2000, 1, 80, 80])
    chaotic_data_40by40 = np.load("Data/TrainingDataProcessed/chaotic_40by40.npy").reshape([4000, 1, 40, 40])
    gaussian_data = np.load("Data/TrainingDataProcessed/gaussian1.npy").reshape([-1, 1, 30, 30])

    training_data = chaotic_data_40by40
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
    ode_train = NeuralODE(ChaoticGaussian40by40Norm2().to(device), ode_solve, step_size).double().to(device)
    # ode_train = torch.load("SavedModels/vortex_conv_gaussian_noTurb.pth")['ode_train']
    n_grid = x_size * y_size  # grid size
    epochs = 10000
    lookahead = 2
    iter_offset = 0
    lr = 0.001
    save_path = "SavedModels/chaotic_conv_gaussian_40by40_normed_noise0_001_4000epochs_model2_1200trainlen"  # file extension will be added in training loop
    train_start_idx = 200  # the index from which training data takes from all data
    train_len = 1100  # length of training data
    step_skip = 6  # number of steps per time interval
    batch_downsize = 7
    obs = torch.tensor(training_data[train_start_idx:train_start_idx + train_len]).view(train_len, nc, x_size,
                                                                                        y_size).double().to(device)
    obs_t = step_skip * torch.tensor((np.arange(len(obs))).astype(int))
    NOISE_VAR = 0.001  # Variance of gaussian noise added to the observation. Assumed to be 0-mean
    obs[:, :, :, :] = obs[:, :, :, :] + torch.randn_like(obs[:, :, :, :]) * NOISE_VAR
    print("Training data shape is: \n", obs.shape)
    # run the model once
    # z_p = ode_train(obs[0].unsqueeze(0), obs_t,
    #                 return_whole_sequence=True)
    # print("z_p size\n", z_p.size())

    # make_color_map(z_p[:, :, :, :].detach().numpy().reshape([train_len, -1]), "Training ConvNN for KS\n",
    #                slice=False, save=None, figure_size=(8, 3))
    # make_color_map(obs[:, :, :, :].detach().cpu().numpy().reshape([train_len, -1]), "Training ConvNN for KS\n",
    #                slice=False, save=None, figure_size=(8, 3))
    # plt.show()

    sample_and_grow_PDE(ode_train, obs, obs_t, epochs, lookahead, iter_offset, lr,
                        save_path, batch_downsize, plot_freq=20)
