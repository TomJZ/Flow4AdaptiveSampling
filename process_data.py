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
    processed_data = np.load("Data/Processed/vortex_re200_no_turbulence.npy")
    V_x = processed_data[:, 0, :, :].reshape(383, -1)
    V_y = processed_data[:, 1, :, :]
    processed_data2 = np.load("Data/Processed/vortex_re200_with_turbulence.npy")
    V_x2 = processed_data2[:, 0, :, :].reshape(383, -1)
    V_y2 = processed_data2[:, 1, :, :]

    diff = V_x-V_x2
    grid = np.load("Data/Processed/regularized_grid.npy")
    X = grid[0, 0, :, :].reshape(-1)
    Y = grid[1, 0, :, :].reshape(-1)

    """
    Generating animation
    """
    anim_data = diff
    SAVE_ANIM = True
    t0 = 100  # the first frame to start animating
    tN = 200  # the last frame to stop animating
    anim = make_flow_anim(X, Y, anim_data, t0=t0, tN=tN,
                          save=SAVE_ANIM,
                          title="Data/Video/turb_noturb_diff2")

    # # save training data
    # training_data = np.stack([V_x_reg, V_y_reg], 1)
    # with open("Data/Processed/vortex_re200_with_turbulence.npy", 'wb') as f:
    #     np.save(f, training_data)
    # print(X_reg.shape)
    # regularized_grid = np.stack([X_reg, Y_reg])
    # with open("Data/Processed/regularized_grid.npy", 'wb') as f:
    #     np.save(f, regularized_grid)

