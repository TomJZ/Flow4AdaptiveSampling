import numpy as np
from tqdm import tqdm
import pandas as pd


# Getting Data for animation
# "/content/drive/Shared drives/ScalAR Lab/MchnLrng_DynSys/data/Wake_Behind_Cylinder/OpenFoamData/Re200_WithTurbulence"
def read_vortex_data(start_idx, end_idx, path):
    start_idx = 200
    end_idx = 384
    vortex_df = []
    X = []
    Y = []
    V_x = []
    V_y = []
    V_z = []
    P = []

    print("Reading Data...")
    for i in tqdm(range(start_idx, end_idx)):
        df = pd.read_csv(path + format(i, '03d') + ".csv")

        x = np.array(df['Points:0'])
        y = np.array(df['Points:1'])
        v_x = np.array(df['U:0'])
        v_y = np.array(df['U:1'])
        v_z = np.array(df['U:2'])
        p = np.array(df['p'])
        index = np.logical_and(np.logical_and(np.logical_and(x > -1.5, y < 5), y > -5), x < 18)
        X.append(x[index])
        Y.append(y[index])
        V_x.append(v_x[index])
        V_y.append(v_y[index])
        V_z.append(v_z[index])
        P.append(p[index])

    X = np.array(X)[0]
    Y = np.array(Y)[0]
    V_x = np.array(V_x)
    V_y = np.array(V_y)
    V_z = np.array(V_z)
    P = np.array(P)

    print("X grid size:\n", X.shape)
    print("Y grid size:\n", Y.shape)
    print("V_x size:\n", V_x.shape)
    print("V_y size:\n", V_y.shape)
    print("V_z size:\n", V_z.shape)
    return X, Y, V_x, V_y, V_z, P
