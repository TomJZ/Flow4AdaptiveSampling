from Utils.GetData import *
from Utils.Plotters import *
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # """
    # Loading Training Data from CSV
    # """
    # start_idx = 0
    # end_idx = 20
    # path = "Data/Re200_WithTurbulence/Re200_Lam_wTurbulence_"
    # X, Y, V_x, V_y, V_z, P = read_vortex_data(start_idx, end_idx, path)
    #
    # # SAVE_PLOT = None
    # # tn = 180  # the nth snapshot, for plotting (verifying purpose only)
    # # plot_vortex(X, Y, V_x, tn=tn, save=SAVE_PLOT)
    # """
    # Making Training Data
    # """
    # # regularize the grid
    # V_x_reg, V_y_reg, X_reg, Y_reg = regularize_flow_data(X, Y, V_x, V_y)
    # plt.imshow(V_x_reg[0].T, interpolation=None, cmap=plt.cm.get_cmap('RdBu_r'), alpha=1)
    # plt.show()

    """
    Loading pre-processed training data from npy
    """
    vortex_noTurb = np.load("Data/Processed/vortex_re200_no_turbulence.npy")
    vortex_V_x = vortex_noTurb[:, 0, :, :].reshape(383, -1)
    vortex_V_y = vortex_noTurb[:, 1, :, :]
    vortex_withTurb = np.load("Data/Processed/vortex_re200_with_turbulence.npy")
    vortex_V_x2 = vortex_withTurb[:, 0, :, :].reshape(383, -1)
    vortex_V_y2 = vortex_withTurb[:, 1, :, :]
    diff = vortex_V_x - vortex_V_x2
    vortex_grid = np.load("Data/Processed/vortex_regularized_grid.npy")
    vortex_X = vortex_grid[0, 0, :, :].reshape(-1)
    vortex_Y = vortex_grid[1, 0, :, :].reshape(-1)

    dg_flow_field = np.load("Data/Processed/dg_flow_field.npy")
    dg_U = dg_flow_field[:, 0, :, :].reshape(200, -1)
    dg_V = dg_flow_field[:, 1, :, :].reshape(200, -1)
    dg_grid = np.load("Data/Processed/dg_grid.npy")
    dg_X = dg_grid[0].reshape(-1)
    dg_Y = dg_grid[1].reshape(-1)

    """
    Generating animation
    """
    anim_data = dg_U
    X = dg_X
    Y = dg_Y
    SAVE_ANIM = True
    t0 = 50  # the first frame to start animating
    tN = 199  # the last frame to stop animating
    anim = make_flow_anim(X, Y, anim_data, t0=t0, tN=tN,
                          save=SAVE_ANIM,
                          title="Data/Video/dg_flow_field")

    # # save training data
    # training_data = np.stack([V_x_reg, V_y_reg], 1)
    # with open("Data/Processed/vortex_re200_with_turbulence.npy", 'wb') as f:
    #     np.save(f, training_data)
    # print(X_reg.shape)
    # regularized_grid = np.stack([X_reg, Y_reg])
    # with open("Data/Processed/vortex_regularized_grid.npy", 'wb') as f:
    #     np.save(f, regularized_grid)
