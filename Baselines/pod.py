from Utils.Plotters import *

def calculate_POD_basis(snapshots, basis_idx=[0], verbose=True):
    '''
    :param snapshots: snapshots imported as 2-D array of size (time len x state dim)
    :basis_idx: array of the index of the sorted basis to be included.

    :return: U, S, V, full reconstructed based on basis, and the POD basis
    '''
    n_time = len(snapshots)
    snapshots = np.transpose(snapshots)

    # singular value decomposition
    U, S, V = np.linalg.svd(snapshots)

    U = U.real
    S = S.real
    V = V.real

    # sorting singular values by magnitude
    I = np.argsort(S)[::-1]
    S = S[I]
    U = U[:, I]

    pod_basis = U[:, basis_idx]
    pod_basis = np.reshape(pod_basis, [-1, pod_basis.shape[1]])

    _, m = pod_basis.shape

    # reconstruction
    S_recon = np.diag(S)[0:m, 0:m]  # only taking the first few to make diagonal matrix.
    V_recon = V[0:m, :]
    full_recon = np.transpose(pod_basis @ S_recon @ V_recon)  # reconstructed data

    if verbose:
        print('#####################\nWith {0} basis\n{1:.3f}% energy is captured\n#####################'.format(m, (
                sum(S[basis_idx]).real / sum(S).real) * 100))
        print('Pod Basis Shape: ', pod_basis.shape)
        print("U shape: ", U.shape)
        print("S shape: ", S.shape, "S recon shape: ", S_recon.shape)
        print("V shape: ", V.shape, "V recon shape: ", V_recon.shape)
        print("reconstruction shape: ", full_recon.shape)
    # pod_basis is the truncated version of U
    return U, S, V, full_recon, pod_basis


if __name__ == "__main__":
    ANIM = False
    SAVE_RECON = True
    vortex_data = np.load("../Data/TrainingDataProcessed/vortex_re200_with_turbulence_flow_field.npy")[:, :, :30, :30]
    vortex_data_long = np.load("../Data/TrainingDataProcessed/vortex_re200_withTurb_long_flow_field.npy")[:, :, :30, :30]
    vortex_grid = np.load("../Data/TrainingDataProcessed/vortex_grid.npy")[:, :30, :30]
    noaa_data = np.load("../Data/TrainingDataProcessed/noaa_flow_field_standard_scaled.npy")
    noaa_grid = np.load("../Data/TrainingDataProcessed/noaa_grid.npy")
    chaotic_data = np.load("../Data/TrainingDataProcessed/chaotic_40by40_vorticity_standard_scaled.npy")
    chaotic_grid = np.load("../Data/TrainingDataProcessed/chaotic_40by40_grid.npy")

    print("Vortex data shape:", vortex_data.shape,
          "\nLong Vortex data shape:", vortex_data_long.shape,
          "\nNOAA data shape: ", noaa_data.shape,
          "\nChaotic data shape: ", chaotic_data.shape)

    data = vortex_data_long
    grid = vortex_grid

    data_len, state_dim, grid_size, _ = data.shape
    pod_input = data.reshape([data_len, -1])
    basis_idx = [0, 1, 2]
    U, S, V, full_recon_raw, pod_basis = calculate_POD_basis(pod_input, basis_idx=basis_idx)

    full_recon = full_recon_raw.reshape([data_len, state_dim, grid_size, grid_size])

    X = grid[0, :, :]
    Y = grid[1, :, :]
    V_x = full_recon[:, 0, :, :]

    t0 = 200
    anim_duration = 40
    anim_save_path = "vortex_pod_3_basis"
    anim_title = "Vortex POD test"

    fig0 = plt.figure(figsize=(10, 10))
    ax0 = fig0.add_subplot(1, 1, 1)
    ax0.set_title("energy distribution of vortex shedding")
    ax0.plot(S, linewidth=0, marker='o')

    low_d = full_recon_raw@pod_basis
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
    ax1.plot(low_d[:, 0], low_d[:, 1], low_d[:, 2])
    ax1.set_title("plot of POD top three energy as coordinates")

    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(1, 1, 1)
    spatial_mode = (pod_basis@S[:len(basis_idx)]).reshape([state_dim, grid_size, grid_size])
    print(spatial_mode.shape)
    ax2.set_title("spatial reconstruction")
    ax2.imshow(spatial_mode[0, :, :])

    if ANIM:
        anim = make_flow_anim(X.reshape(-1), Y.reshape(-1), V_x.reshape(data_len, -1), t0=t0, tN=t0+anim_duration,
                              save_path=anim_save_path,
                              title=anim_title)
    if SAVE_RECON:
        with open("SavedModel/vortex_long_pod_3_basis.npy", "wb") as f:
            np.save(f, full_recon)

    plt.show()
