import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import griddata


def read_vortex_data(start_idx, end_idx, path):
    """
    Args:
        start_idx: int, the time index to start reading from
        end_idx: int, the time index to end reading
        path: the file path, except for the number and file extension

    Returns:

    """
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


def regularize_grid(X, Y, V_x):
    """
      Regularize an irregular grid by interpolation
      Input:
        X: simulation grid X coordinate values
        Y: simulation grid Y coordinate values
        V_x: the data to be regularized
      Return:
        grid_x: regularized X coordinates
        grid_y: regularized Y coordinates
        V_x_regular: regularized data
    """
    grid_x, grid_y = np.mgrid[2:18:54j, -5:5:32j]
    irregular_coor = np.concatenate([X.reshape([-1, 1]), Y.reshape(-1, 1)], 1)
    values = V_x
    V_x_regular = griddata(irregular_coor, values, (grid_x, grid_y), method='linear')
    return grid_x, grid_y, V_x_regular


def regularize_flow_data(X, Y, V_x, V_y):
    data_len = len(V_x)
    # creating regularized data
    V_x_reg = []
    V_y_reg = []
    X_list = []
    Y_list = []

    print("Regularizing Grid...")
    for snapshot_t in tqdm(range(data_len - 1)):  # only taking the 200th to the 250th step data
        X_reg, Y_reg, curr_V_x = regularize_grid(X, Y, V_x[snapshot_t])
        X_reg, Y_reg, curr_V_y = regularize_grid(X, Y, V_y[snapshot_t])
        V_x_reg.append(curr_V_x)
        V_y_reg.append(curr_V_y)
        X_list.append(X_reg)
        Y_list.append(Y_reg)

    V_x_reg = np.array(V_x_reg)
    V_y_reg = np.array(V_y_reg)
    X_reg = np.array(X_list)
    Y_reg = np.array(Y_list)

    # remove uninterpolated borders
    # chopping away uninterpolated points (i.e. remove the borders)
    V_x_reg = V_x_reg[:, 2:-2, 1:-1]
    X_reg = X_reg[:, 2:-2, 1:-1]
    V_y_reg = V_y_reg[:, 2:-2, 1:-1]
    Y_reg = Y_reg[:, 2:-2, 1:-1]

    return V_x_reg, V_y_reg, X_reg, Y_reg


def get_oscillating_cylinder_data(start_idx, end_idx, path):
    """
    Args:
        start_idx: int, the time index to start reading from
        end_idx: int, the time index to end reading
        path: the file path, except for the number and file extension

    Returns:

    """
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
        index = np.logical_and(np.logical_and(np.logical_and(x > -1.5, y < 5), y > -5), x < 18)

        x_reg, y_reg, v_x_reg = regularize_grid(x[index], y[index], v_x[index])
        x_reg, y_reg, v_y_reg = regularize_grid(x[index], y[index], v_y[index])

        X.append(x_reg[2:-2, 1:-1])
        Y.append(y_reg[2:-2, 1:-1])
        V_x.append(v_x_reg[2:-2, 1:-1])
        V_y.append(v_y_reg[2:-2, 1:-1])

    X = np.array(X)[0]
    Y = np.array(Y)[0]
    V_x = np.array(V_x)
    V_y = np.array(V_y)

    print("X grid size:\n", X.shape)
    print("Y grid size:\n", Y.shape)
    print("V_x size:\n", V_x.shape)
    print("V_y size:\n", V_y.shape)
    return X, Y, V_x, V_y