from Utils.GetData import *
from Utils.Plotters import *
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_idx = 0
    end_idx = 384
    path = "Data/Re200_NoTurbulence/Re200_Lam_NoTurb_"
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
    with open("Data/Processed/vortex_re200_no_turbulence.npy", 'wb') as f:
        np.save(f, training_data)
