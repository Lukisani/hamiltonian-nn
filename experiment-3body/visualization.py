# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# If needed, import the same data/model functions used in test
# from test import get_args, ObjectView, random_config, get_orbit, load_model, model_update
# from data import potential_energy, kinetic_energy, total_energy
# Or load pickled data from disk as needed.

def plot_trajectories(orbit, base_orbit=None, hnn_orbit=None, settings=None, fig_dir='./figures', format='pdf'):
    """Plot trajectories and energy comparisons."""
    fig = plt.figure(figsize=[15,4], dpi=100)
    lw = 3
    fs=9
    ts=15
    tpad = 7
    ls=12

    # Ground truth orbit
    plt.subplot(1,3,1)
    plt.title('Trajectories', fontsize=ts, pad=tpad)
    colors = ['orange', 'purple', 'blue']
    for i, path in enumerate(orbit):
        plt.plot(path[1], path[2], '-', c=colors[i], label='True path, body {}'.format(i), linewidth=2)

    if base_orbit is not None:
        for i, path in enumerate(base_orbit):
            plt.plot(path[1], path[2], '--', c=colors[i], label='NN path, body {}'.format(i), linewidth=2)

    plt.axis('equal')
    plt.xlabel('$x$', fontsize=ls)
    plt.ylabel('$y$', fontsize=ls)
    plt.legend(fontsize=fs)

    if settings is not None:
        # Plot Ground truth energy
        plt.subplot(1,3,2)
        real_pe = potential_energy(orbit)
        real_ke = kinetic_energy(orbit)
        real_etot = total_energy(orbit)
        plt.title('Ground truth energy', fontsize=ts, pad=tpad)
        plt.xlabel('Time')
        plt.plot(settings['t_eval'], real_pe, 'g:', label='Potential', linewidth=lw)
        plt.plot(settings['t_eval'], real_ke, 'c-.', label='Kinetic', linewidth=lw)
        plt.plot(settings['t_eval'], real_etot, 'k-', label='Total', linewidth=lw)
        plt.legend(fontsize=fs)
        plt.xlim(*settings['t_span'])
        ymin = np.min([real_pe.min(), real_ke.min(), real_etot.min()])
        ymax = np.max([real_pe.max(), real_ke.max(), real_etot.max()])
        plt.ylim(ymin, ymax)

        # Plot baseline/HNN energy if provided
        if base_orbit is not None:
            plt.subplot(1,3,3)
            plt.title('Baseline NN energy', fontsize=ts, pad=tpad)
            plt.xlabel('Time')
            plt.plot(settings['t_eval'], potential_energy(base_orbit), 'g:', label='Potential', linewidth=lw)
            plt.plot(settings['t_eval'], kinetic_energy(base_orbit), 'c-.', label='Kinetic', linewidth=lw)
            plt.plot(settings['t_eval'], total_energy(base_orbit), 'k-', label='Total', linewidth=lw)
            plt.legend(fontsize=fs)
            plt.xlim(*settings['t_span'])
            plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()
    fig.savefig('{}/3body-compare.{}'.format(fig_dir, format))

def plot_training_curves(base_stats, hnn_stats, fig_dir='./figures', format='pdf'):
    """Plot the training curves for baseline and HNN."""
    import scipy.ndimage
    gaussian_filter = scipy.ndimage.gaussian_filter

    fw = 200
    fig = plt.figure(figsize=[5,3], dpi=300)
    plt.plot(gaussian_filter(hnn_stats['test_loss'], fw), 'b-', label='hnn-test')
    plt.plot(gaussian_filter(hnn_stats['train_loss'], fw), 'b--', label='hnn-train')
    plt.plot(gaussian_filter(base_stats['test_loss'], fw), 'r-', label='base-test')
    plt.plot(gaussian_filter(base_stats['train_loss'], fw), 'r--', label='base-train')

    plt.title('3-body training curves')
    plt.xlabel('Gradient step'); plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig('{}/3body-train-curves-long.{}'.format(fig_dir, format))


# Additional visualization functions like plot_orbits for GIFs could also be added here.

