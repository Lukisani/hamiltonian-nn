# For testing efficacy of models (using MSE for now)

# -*- coding: utf-8 -*-

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PARENT_DIR)

import torch, sys
import numpy as np
import scipy.integrate, scipy.ndimage
from nn_models import MLP
from hnn import HNN
from utils import L2_loss, to_pickle, from_pickle
from data3d import get_dataset, get_orbit, random_config
from data3d import potential_energy, kinetic_energy, total_energy

solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './experiment-3body'
sys.path.append(EXPERIMENT_DIR)

class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d

def get_args():
    return {
        'input_dim': 3*4, # two bodies, each with q_x, q_y, p_x, p_y
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
        'fig_dir': './figures'
    }

def load_model(args, baseline=False):
    output_dim = args.input_dim if baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=baseline)
    case = 'baseline' if baseline else 'hnn'
    path = "{}/{}-orbits-{}.tar".format(args.save_dir, args.name, case)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

def model_update(t, state, model):
    """Given time t and state, use the model to compute time derivatives."""
    state = state.reshape(-1,5)
    deriv = np.zeros_like(state)
    np_x = state[:,1:] # drop mass
    np_x = np_x.T.flatten()[None, :]
    x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_derivative(x)
    deriv[:,1:] = dx_hat.detach().data.numpy().reshape(4,3).T
    return deriv.reshape(-1)

if __name__ == "__main__":
    # Example usage: Load models and run test trajectories
    args = ObjectView(get_args())
    np.random.seed(args.seed)

    # Load baseline and HNN models
    base_model = load_model(args, baseline=True)
    hnn_model = load_model(args, baseline=False)

    # Run a single test trajectory and compute energies
    state = random_config()
    orbit, settings = get_orbit(state, t_points=2000, t_span=[0,5])
    base_update_fn = lambda t, y0: model_update(t, y0, base_model)
    hnn_update_fn = lambda t, y0: model_update(t, y0, hnn_model)

    base_orbit, _ = get_orbit(state, t_points=2000, t_span=[0,5], update_fn=base_update_fn)
    hnn_orbit, _ = get_orbit(state, t_points=2000, t_span=[0,5], update_fn=hnn_update_fn)

    # Compute total energy differences as a basic "test" metric
    real_etot = total_energy(orbit)
    base_etot = total_energy(base_orbit)
    hnn_etot = total_energy(hnn_orbit)

    base_dist = (real_etot - base_etot)**2
    hnn_dist = (real_etot - hnn_etot)**2

    print("Baseline NN energy MSE:", np.mean(base_dist))
    print("HNN energy MSE:", np.mean(hnn_dist))
