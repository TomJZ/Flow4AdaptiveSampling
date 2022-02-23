import time
from Utils.Plotters import *
from NODE.NODE import *

if __name__ == "__main__":
    """
    load pretrained model
    """
    ode_train = torch.load("SavedModels/vortex_conv_gaussian.pth",
                           map_location=torch.device('cpu'))['ode_train']

    """
    Load initial condition from data 
    """
    snapshot = 200  # the snapshot to use as initial condition
    processed_data = np.load("Data/Processed/vortex_re200_no_turbulence.npy")
    grid = np.load("Data/Processed/regularized_grid.npy")
    X = grid[0, 0, :, :]
    Y = grid[1, 0, :, :]
    initial_condition = torch.tensor(processed_data[snapshot]).unsqueeze(0)
    print(initial_condition.shape)

    """
    Generating predictions
    """
    test_len = 50
    step_skip = 5
    start = time.time()
    z_p = ode_train(initial_condition,
                    step_skip * torch.tensor(np.arange(test_len).astype(int)),
                    return_whole_sequence=True)
    duration = time.time() - start
    print("Total time taken to make prediction is: ", duration, " seconds")
    pred = z_p.detach().numpy()[:, 0, :, :]
    print("Prediction has shape: ", pred.shape)
    V_x = pred[:, 0, :, :]
    SAVE_ANIM = True
    SAVE_PLOT = None
    t0 = 0
    tN = t0 + test_len

    print(X.shape, V_x.shape)
    anim = make_flow_anim(X.reshape(-1), Y.reshape(-1), V_x.reshape(test_len, -1), t0=t0, tN=tN,
                          save=SAVE_ANIM,
                          title="Data/Video/prediction_noTurb")