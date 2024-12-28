import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch, time, sys
import numpy as np
import scipy.integrate, scipy.ndimage
solve_ivp = scipy.integrate.solve_ivp
gaussian_filter = scipy.ndimage.gaussian_filter

EXPERIMENT_DIR = './experiment-3body'
sys.path.append(EXPERIMENT_DIR)

# from nn_models import MLP
# from hnn import HNN
from utils import L2_loss, to_pickle, from_pickle
from data import get_dataset, get_orbit, random_config
from data import potential_energy, kinetic_energy, total_energy

DPI = 300
FORMAT = 'pdf'

def get_args():
    return {'input_dim': 3*4, # two bodies, each with q_x, q_y, p_z, p_y
         'hidden_dim': 200,
         'learn_rate': 1e-3,
         'input_noise': 0.,
         'batch_size': 600,
         'nonlinearity': 'tanh',
         'total_steps': 1500,
         'field_type': 'solenoidal',
         'print_every': 200,
         'verbose': True,
         'name': '3body',
         'seed': 0,
         'save_dir': '{}'.format(EXPERIMENT_DIR),
         'fig_dir': './figures'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

def plot_groundtruth():
    args = ObjectView(get_args())
    np.random.seed(0)
    state = random_config()
    orbit, settings = get_orbit(state, t_points=1000, t_span = [0, 5], rtol = 1e-9)

    # draw trajectories
    fig = plt.figure(figsize=[10,4], dpi=100)
    plt.subplot(1,2,1)
    plt.title('Trajectories')
    for i, path in enumerate(orbit):
        plt.plot(path[1], path[2], label='body {} path'.format(i))

    plt.axis('equal')
    plt.xlabel('$x$') ; plt.ylabel('$y$')
    plt.legend(fontsize=8)

    plt.subplot(1,2,2)
    plt.title('Energy') ; plt.xlabel('time')
    plt.plot(settings['t_eval'], potential_energy(orbit), label='potential')
    plt.plot(settings['t_eval'], kinetic_energy(orbit), label='kinetic')
    plt.plot(settings['t_eval'], total_energy(orbit), label='total')
    plt.legend()
    plt.xlim(*settings['t_span'])

    plt.show()
    fig.savefig('{}/orbits-dataset.{}'.format(args.fig_dir, FORMAT))

plot_groundtruth()