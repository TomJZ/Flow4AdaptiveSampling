import time
from Utils.Plotters import *
from Utils.POD import *
from NODE.NODE import *

if __name__ == "__main__":
    flow = 'chaotic'  # 'dg' for double gyre, 'vortex' for vortex shedding, 'noaa' for ocean data, 'chaotic' for forced turbulence, 'gaussian' for gaussian blobs
    model_path = "SavedModels/chaotic_conv_gaussian_40by40_normed_noise0_001_18000epochs_model2_1100trainlen.pth"
    training_data_path = "Data/Processed/chaotic_40by40.npy"
    init_con_snapshot = 1100
    grid_path = "Data/Processed/chaotic_grid_40by40.npy"
    test_len = 400  # length of prediction to generate
    step_skip = 6  # number of steps within one time interval
    anim_save_path = "Data/Video/chaotic_40by40_pred"
    anim_title = "training prediction chaotic vorticity"
    pred_save_path = "Data/Predictions/chaotic_1100_to_1500.npy"
    square = True  # if only looks at the square area in vortex shedding
    generate_animation = False  # whether to generate animation and save
    generate_POD = False  # whether to compute POD energies
    save_prediction = True  # whether to save predicted trajectories

    if flow == 'dg':
        data_shrink_scale = 2
    elif flow == 'vortex' or flow == 'noaa' or flow == 'chaotic' or flow == 'gaussian':
        data_shrink_scale = 1
    else:
        print("There is no such flow type!")

    """
    load pretrained model
    """
    ode_train = torch.load(model_path, map_location=torch.device('cpu'))['ode_train']

    """
    Load initial condition from data 
    """
    processed_data = np.load(training_data_path).reshape([-1, 1, 40, 40])[:, :, ::data_shrink_scale, ::data_shrink_scale]
    grid = np.load(grid_path)[:, ::data_shrink_scale, ::data_shrink_scale]
    if flow == 'dg':
        # cropping double gyre to a square
        processed_data = processed_data[:, :, :50, :50]
        grid = grid[:, :50, :50]
    elif flow == 'noaa' or flow == 'chaotic' or flow == 'gaussian':
        # noaa data is already cropped to a square
        processed_data = processed_data.astype(np.double)
    elif flow == 'vortex':
        if square:
            processed_data = processed_data[:, :, :30, :30]
            grid = grid[:, :30, :30]

    X = grid[0, :, :]
    Y = grid[1, :, :]
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
    if flow != 'chaotic' and flow != 'gaussian':
        V_y = pred[:, 1, :, :]
        mag = np.sqrt(V_x ** 2 + V_y ** 2)

    """
    Animating Predictions
    """
    if generate_animation:
        SAVE_ANIM = True
        SAVE_PLOT = None
        t0 = 0
        tN = t0 + test_len

        anim = make_flow_anim(X.reshape(-1), Y.reshape(-1), V_x.reshape(test_len, -1), t0=t0, tN=tN,
                              save_path=anim_save_path,
                              title=anim_title)

    """
    Saving Predictions
    """
    if save_prediction:
        with open(pred_save_path, 'wb') as f:
            np.save(f, pred)

    """
    Generating POD basis
    """
    if generate_POD:
        _, energies, _, _, _ = calculate_POD_basis(pred.reshape(test_len, -1))
        plt.scatter(np.arange(len(energies)), energies)
        plt.show()
