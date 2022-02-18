from Utils.GetData import *
from Utils.Plotters import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_idx = 200
    end_idx = 385
    path = "C:/Users/tom_j/Downloads/Re200_NoTurbulence/Re200_Lam_NoTurb_"
    X, Y, V_x, V_y, V_z, P = read_vortex_data(start_idx, end_idx, path)

    SAVE_PLOT = None
    tn = 180  # the nth snapshot, for plotting (verifying purpose only)
    plot_vortex(X, Y, V_x, tn=tn, save=SAVE_PLOT)

    SAVE_ANIM = True
    t0 = 0  # the first frame to start animating
    tN = 100  # the last frame to stop animating
    anim = make_flow_anim(X, Y, V_x, t0=t0, tN=tN, save=SAVE_ANIM, title="VortexShedding_Re40_Laminar")
