import time
from Utils.Plotters import *
from NODE.NODE import *

if __name__ == "__main__":
    """
    load pretrained model
    """
    ode_train = torch.load("SavedModels/vortex_conv_gaussian_noTurb.pth",
                           map_location=torch.device('cpu'))['ode_train']

    """
    Load initial condition from data 
    """
    snapshot = 300  # the snapshot to use as initial condition
    processed_data = np.load("Data/Processed/vortex_re200_no_turbulence.npy")
    grid = np.load("Data/Processed/vortex_regularized_grid.npy")
    X = grid[0, 0, :, :]
    Y = grid[1, 0, :, :]
    initial_condition = torch.tensor(processed_data[snapshot]).unsqueeze(0)
    print("Initial condition has shape: ", initial_condition.shape)

    """
    Generating predictions
    """
    test_len = 100  # length of prediction to generate
    step_skip = 6  # number of steps within one time interval
    start = time.time()
    # generating predictions
    z_p = ode_train(initial_condition,
                    step_skip * torch.tensor(np.arange(test_len).astype(int)),
                    return_whole_sequence=True)
    duration = time.time() - start
    pred = z_p.detach().numpy()[:, 0, :, :]
    print("Total time taken to make prediction is: ", duration, " seconds")
    print("Prediction has shape: ", pred.shape)
    # predicted V_x and V_y
    V_x = pred[:, 0, :, :]
    V_y = pred[:, 1, :, :]

    """
    Animating Predictions
    """
    SAVE_ANIM = True
    SAVE_PLOT = None
    t0 = 0
    tN = t0 + test_len

    anim = make_flow_anim(X.reshape(-1), Y.reshape(-1), V_x.reshape(test_len, -1), t0=t0, tN=tN,
                          save=SAVE_ANIM,
                          title="Data/Video/prediction_noTurb")