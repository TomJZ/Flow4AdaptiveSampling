import

def sample_and_grow_PDE(ode_train, true_sampled_traj, true_sampled_times, epochs, lookahead, plot_freq=300):
    optimizer = torch.optim.Adam(ode_train.parameters(), lr=LR)
    n_segments = len(true_sampled_traj)
    obs_list = []
    for i in range(len(true_sampled_traj) - lookahead):
        obs_list.append(true_sampled_traj[i:i + lookahead].unsqueeze(1))
    obs_ = torch.cat(obs_list, 1).squeeze()

    for i in range(epochs):
        # Train Neural ODE
        init_conditions = true_sampled_traj[:-lookahead]  # initial conditions to start integrating
        duration = true_sampled_times[:lookahead]  # we assume regularly sampled data here
        preds = ode_train(init_conditions, duration, return_whole_sequence=True)
        preds = preds.squeeze()
        # print("new pred shape: ", preds.shape)
        # print("observation shape:", obs_.shape)
        # print("++++++++++++++++ pred:\n",preds, "+++++++++++++++ obs:\n", obs_)

        loss = F.mse_loss(preds, obs_)
        loss_arr.append(loss.item())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % plot_freq == 0:
            print("Plotting...")
            updated_title = "Iteration: {0} Loss: {1:.3e}\n No. of Points: {2} Lookahead: {3} LR: {4}".format(
                i + ITER_OFFSET, loss.item(), len(true_sampled_traj), lookahead - 1, LR)
            CHECKPOINT_PATH = save_path + "/Vortex_biconv_gaussian.pth"
            torch.save({'ode_train': ode_train, 'loss_arr': loss_arr}, CHECKPOINT_PATH)
            z_p = ode_train(true_sampled_traj[0].unsqueeze(0), true_sampled_times, return_whole_sequence=True)
            make_color_map(z_p[:, :, :, :].cpu().detach().numpy().reshape([TRAIN_LEN, -1]),
                           "Training NN for Vortex\n" + updated_title,
                           slice=False, save=save_path + name + f"_color_{i + ITER_OFFSET}.png", figure_size=(10, 10))
            clear_output(wait=True)