from Utils.GetData import *
from Utils.Plotters import *
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # """
    # Loading Training Data from CSV
    # """
    # start_idx = 0
    # end_idx = 399
    # path = "Data/Raw/Re200_Oscillating/Re200_Oscillating_"
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
    # X_reg, Y_reg, V_x_reg, V_y_reg = get_oscillating_cylinder_data(start_idx, end_idx, path)
    # """
    # Loading pre-processed training data from npy
    # """
    # vortex_noTurb = np.load("Data/Processed/vortex_re200_no_turbulence.npy")
    # vortex_V_x = vortex_noTurb[:, 0, :, :].reshape(383, -1)
    # vortex_V_y = vortex_noTurb[:, 1, :, :].reshape(383, -1)
    # vortex_withTurb = np.load("Data/Processed/vortex_re200_with_turbulence.npy")
    # vortex_V_x2 = vortex_withTurb[:, 0, :, :].reshape(383, -1)
    # vortex_V_y2 = vortex_withTurb[:, 1, :, :].reshape(383, -1)
    # diff = vortex_V_x - vortex_V_x2
    vortex_grid = np.load("Data/Processed/vortex_regularized_grid.npy")
    vortex_X = vortex_grid[0, :, :].reshape(-1)
    vortex_Y = vortex_grid[1, :, :].reshape(-1)
    #
    # mag = np.sqrt(vortex_V_x2**2 + vortex_V_y2**2)
    #
    #
    dg_flow_field = np.load("Data/Processed/dg_flow_field.npy")
    dg_flow_field = dg_flow_field[:, :, ::2, ::2]
    print(dg_flow_field.shape)
    dg_U = dg_flow_field[:, 0, :50, :50].reshape(200, -1)
    dg_V = dg_flow_field[:, 1, :50, :50].reshape(200, -1)
    print(dg_U.shape)
    dg_grid = np.load("Data/Processed/dg_grid.npy")
    dg_grid = dg_grid[:, ::2, ::2]
    print(dg_grid.shape)
    dg_X = dg_grid[0, :50, :50].reshape(-1)
    dg_Y = dg_grid[1, :50, :50].reshape(-1)
    print(dg_X.shape)
    #
    """
    Generating animation
    """
    anim_data = dg_U
    X = dg_X
    Y = dg_Y
    save_path = "Data/Video/dg_square_anim"
    t0 = 0  # the first frame to start animating
    tN = 50  # the last frame to stop animating
    anim = make_flow_anim(X.reshape(-1), Y.reshape(-1), anim_data.reshape(200, -1), t0=t0, tN=tN,
                          save_path=save_path,
                          title="dg square")

    # # save training data
    # regularized_grid = np.stack([X_reg, Y_reg])
    # print("grid shape: ", regularized_grid.shape)
    # with open("Data/Processed/vortex_regularized_grid.npy", 'wb') as f:
    #     np.save(f, regularized_grid)
    # training_data = np.stack([V_x_reg, V_y_reg], 1)
    # with open("Data/Processed/vortex_re200_oscillating.npy", 'wb') as f:
    #     np.save(f, training_data)

    # with open("Data/Processed/training_mag_withTurb.npy", 'wb') as f:
    #      np.save(f, mag.reshape(383, 50, 30))
