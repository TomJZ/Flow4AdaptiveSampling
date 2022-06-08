from torch.nn import functional as F
from Utils.Plotters import *
import torch
import numpy as np
import tqdm


def sample_and_grow_PDE(ode_train, true_sampled_traj, true_sampled_times, epochs, lookahead, ITER_OFFSET, LR,
                        save_path, batch_downsize, plot_freq=300):
    optimizer = torch.optim.Adam(ode_train.parameters(), lr=LR)
    train_len = len(true_sampled_traj)
    obs_list = []
    for i in range(len(true_sampled_traj) - lookahead):
        obs_list.append(true_sampled_traj[i:i + lookahead].unsqueeze(1))
    obs_ = torch.cat(obs_list, 1).squeeze()
    loss_arr = []

    for i in tqdm.tqdm(range(epochs)):
        # Train Neural ODE
        init_conditions = true_sampled_traj[:-lookahead]  # initial conditions to start integrating
        total_size = init_conditions.shape[0]
        #print("init condition shape", init_conditions.shape)
        batch_idx = np.arange(total_size)[i%batch_downsize::batch_downsize]
        #print("batch idx is: \n", batch_idx)
        duration = true_sampled_times[:lookahead]  # we assume regularly sampled data here
        preds = ode_train(init_conditions[batch_idx], duration, return_whole_sequence=True)
        preds = preds.squeeze()
        #print("pred and obs sizes: ", preds.shape, obs_.shape)
        loss = F.mse_loss(preds[1], obs_[1, batch_idx])
        loss_arr.append(loss.item())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % plot_freq == 0:
            updated_title = "Iteration: {0} Loss: {1:.3e}\n No. of Points: {2} Lookahead: {3} LR: {4}".format(
                i + ITER_OFFSET, loss.item(), len(true_sampled_traj), lookahead - 1, LR)
            print(updated_title)
            if save_path is not None:
                CHECKPOINT_PATH = save_path + ".pth"
                torch.save({'ode_train': ode_train, 'loss_arr': loss_arr}, CHECKPOINT_PATH)
                img_save_path = save_path + f"_color_{i + ITER_OFFSET}.png"
            else:
                img_save_path = None

            z_p = ode_train(true_sampled_traj[0].unsqueeze(0), true_sampled_times, return_whole_sequence=True)
            #make_color_map(z_p[:, :, :, :].cpu().detach().numpy().reshape([train_len, -1]),
            #               "Training NN for Vortex\n" + updated_title,
            #               slice=False, save=img_save_path, figure_size=(10, 5))
            #plt.show()
