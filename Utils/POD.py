import numpy as np
import matplotlib as mpl

def calculate_POD_basis(snapshots, basis_idx=[0], verbose=True):
    '''
    :param snapshots: snapshots imported as 2-D array of size (time len x state dim)
    :param basis_idx: array of the indices of the sorted basis to be included, 0 being the basis with the most energy
    :param verbose: set to True if to print details
    :return U: state dynamics
    :return S: eigenvalues
    :return V: time dynamics
    :return full_recon: full data reconstructed based on basis included
    :return pod_basis: the POD basis
    '''

    if len(snapshots.shape) == 3:
        snapshots = snapshots.reshape(snapshots.shape[0], -1)
    elif len(snapshots.shape) == 2:
        pass

    n_time = len(snapshots)
    snapshots = np.transpose(snapshots)
    # snapshots = snapshots.T@snapshots

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
        print("U shape:\n", U.shape)
        print("S shape:\n", S.shape)
        print("V shape:\n", V.shape)
        print("reconstruction shape:\n", full_recon.shape)
    return U, S, V, full_recon, pod_basis


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    noaa_path = "../Data/TrainingDataProcessed/noaa_30by30_flow_field_standard_scaled.npy"
    vortex_path = "../Data/TrainingDataProcessed/vortex_re200_withTurb_long_flow_field.npy"
    data = np.load(vortex_path)[300:350, :30, :30]
    data_len = data.shape[0]
    print("data shape is: ", data.shape)
    _, energies, _, _, _ = calculate_POD_basis(data.reshape(data_len, -1), basis_idx=[0, 1, 2, 3])
    mpl.rcParams.update({'font.size': 15})
    plt.figure(figsize=(6, 3))
    plt.scatter(np.arange(len(energies)), energies)
    plt.show()
