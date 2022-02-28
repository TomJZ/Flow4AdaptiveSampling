import time
from Utils.Plotters import *
from NODE.NODE import *

if __name__ == "__main__":
    model_path = "SavedModels/dg_conv_gaussian_small.pth"
    training_data_path = "Data/Processed/dg_flow_field.npy"
    init_con_snapshot = 0
    grid_path = "Data/Processed/dg_grid.npy"
    test_len = 100  # length of prediction to generate
    step_skip = 4  # number of steps within one time interval
    anim_save_path = "Data/Video/prediction3_dg_on_training_data"
    anim_title = "training prediction3 dg"
    data_shrink_scale = 3

    """
    load pretrained model
    """
    ode_train = torch.load(model_path, map_location=torch.device('cpu'))['ode_train']

    """
    Load initial condition from data 
    """
    processed_data = np.load(training_data_path)[:, :, ::data_shrink_scale, ::data_shrink_scale]
    grid = np.load(grid_path)
    X = grid[0, ::data_shrink_scale, ::data_shrink_scale]
    Y = grid[1, ::data_shrink_scale, ::data_shrink_scale]
    initial_condition = torch.tensor(processed_data[init_con_snapshot]).unsqueeze(0)
    print("Initial condition has shape: ", initial_condition.shape)

    """
    Generating predictions
    """

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
    pred_mag = np.sqrt(V_x ** 2 + V_y ** 2)

    """
    Animating Predictions
    """
    SAVE_ANIM = True
    SAVE_PLOT = None
    t0 = 0
    tN = t0 + test_len

    anim = make_flow_anim(X.reshape(-1), Y.reshape(-1), V_x.reshape(test_len, -1), t0=t0, tN=tN,
                          save_path=anim_save_path,
                          title=anim_title)

    print("pred mag shape is: ", pred_mag.shape)
    # with open("Data/Processed/prediction_oscillating_Vx.npy", 'wb') as f:
    #      np.save(f, pred_mag)