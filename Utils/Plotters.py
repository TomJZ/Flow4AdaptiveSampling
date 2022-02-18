import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from matplotlib import animation, rcParams


def make_color_map(DATA_TO_PLOT, title="Untitled", slice=False, save=None, figure_size=(10, 3)):
    mpl.rcParams.update({'font.size': 7})
    nsteps, ngrids, *a = DATA_TO_PLOT.shape

    offset = 0  # number of grids (starting from 0th) not to plot

    # creating mesh
    x = np.linspace(0, nsteps - 1, num=nsteps)
    y = np.linspace(0, ngrids - offset - 1, num=ngrids - offset)
    X, Y = np.meshgrid(x, y)

    # plotting color grid
    fig = plt.figure(figsize=figure_size)
    ax = plt.axes()
    c = ax.pcolormesh(X, Y, DATA_TO_PLOT.T.reshape(ngrids, nsteps)[offset:ngrids, :], cmap="jet")
    ax.set_title(title + '\nHeat Map')
    ax.set_xlabel('time')  # adding axes labels changes the appearance of the color map
    ax.set_ylabel('space')
    fig.tight_layout()
    fig.colorbar(c)
    if save is not None:
        plt.savefig(save)
    plt.show()

    if slice:
        # plotting one single trajectory
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        # for i in range(12):
        # i = i*10
        ax.plot(DATA_TO_PLOT[0, :])
        ax.plot(DATA_TO_PLOT[1, :])
        ax.set_title(title + '\nSingle Trajectory')
        ax.set_xlabel('time')
        ax.set_ylabel('state')
        fig.tight_layout()
        plt.show()


def plot_vortex(X, Y, V, tn, save=None, scatter_coor=None):
    """
    X: Flattened 1D array from a 2D meshgrid
    Y: Flattened 1D array from a 2D meshgrid
    V: [time_len, -1] the second dimension must match the size of the flattened meshgrid
    tn: the snapshot to plot
    """
    n_levels = 500
    mpl.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(7, 3), nrows=1)
    cntr = ax.tricontourf(X[:], Y[:], V[tn], levels=n_levels, cmap="RdBu_r")
    cb = fig.colorbar(cntr, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=6)
    cb.locator = tick_locator
    cb.update_ticks()
    # ax.set(xlim=(-1.5, 17.5), ylim=(-4.5, 4.5))
    ax.set_xlabel("x", labelpad=2)
    ax.set_ylabel("y", labelpad=2)
    plt.subplots_adjust(hspace=0.5)

    if scatter_coor is not None:
        ax.scatter(scatter_coor[:, 0], scatter_coor[:, 1], c='green', marker='.', s=10)

    if save is not None:
        plt.savefig(save + '.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0)
        plt.savefig(save + '.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0)
        pass
    plt.show()


def make_flow_anim(X, Y, V, t0, tN, save=False, title=''):
    fig = plt.figure(figsize=(7, 3.7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('black')
    n_levels = 500
    c = ax.tricontourf(X[:], Y[:], V[t0], levels=n_levels, cmap="RdBu_r")
    cb = fig.colorbar(c, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=6)
    ax.set_title('Vortex Shedding Vx')
    # ax.set(xlim=(-1.5, 17.5), ylim=(-4.5, 4.5))
    fig.tight_layout()
    rcParams['animation.embed_limit'] = 2 ** 128
    img_list = []

    for i in range(t0, tN, 1):
        if i % 10 == 0:
            print("Animating frame: ", i)
        c = ax.tricontourf(X[:], Y[:], V[i], levels=n_levels, cmap="RdBu_r")
        img_list.append(c.collections)

    anim = animation.ArtistAnimation(fig, img_list, interval=50)

    if save:
        anim.save(title + '.mp4', writer=animation.FFMpegWriter(fps=12))

    return anim
