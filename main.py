from Utils.GetData import *
from Utils.Plotters import *
from Utils.Solvers import *
from NODE.NODE import *
from models import *
import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_idx = 200
    end_idx = 385
    path = "Data/Re200_WithTurbulence/Re200_Lam_wTurbulence_"
    X, Y, V_x, V_y, V_z, P = read_vortex_data(start_idx, end_idx, path)

    # SAVE_PLOT = None
    # tn = 180  # the nth snapshot, for plotting (verifying purpose only)
    # plot_vortex(X, Y, V_x, tn=tn, save=SAVE_PLOT)
    #
    # SAVE_ANIM = True
    # t0 = 0  # the first frame to start animating
    # tN = 100  # the last frame to stop animating
    # anim = make_flow_anim(X, Y, V_x, t0=t0, tN=tN, save=SAVE_ANIM, title="VortexShedding_Re40_Laminar")

    """
    Making Training Data
    """
    # regularize the grid
    V_x_reg, V_y_reg, _, _ = regularize_flow_data(X, Y, V_x, V_y)
    plt.imshow(V_x_reg[0].T, interpolation=None, cmap=plt.cm.get_cmap('RdBu_r'), alpha=1)
    plt.show()

    # training data
    training_data = np.stack([V_x_reg, V_y_reg], 1)

    """
    Training
    """
    Hybrid = False
    ode_solve = Euler
    step_size = 0.01
    loss_arr = []  # initializing loss array
    ode_train = NeuralODE(Vortex_biconv_gaussian())
    n_grid = 1500
    vortex_train = training_data
    train_len = 50
    obs_notime = torch.tensor(vortex_train).view(train_len, 2, 50, 30)
    obs = obs_notime
    NOISE_VAR = 0.0  # Variance of gaussian noise added to the observation. Assumed to be 0-mean
    obs[:, :, :, 0:2] = obs[:, :, :, 0:2] + torch.randn_like(obs[:, :, :, 0:2]) * NOISE_VAR

    print("observation size\n", obs.size())
    z_p = ode_train(obs[0], 4 * torch.tensor((np.arange(len(obs))).astype(int)), return_whole_sequence=True)
    print("z_p size\n", z_p.size())

    make_color_map(z_p[:, :, :, :].detach().numpy().reshape([train_len, -1]), "Training ConvNN for KS\n",
                   slice=False, save=None, figure_size=(8, 3))
    make_color_map(obs[:, :, :, :].detach().numpy().reshape([train_len, -1]), "Training ConvNN for KS\n",
                   slice=False, save=None, figure_size=(8, 3))