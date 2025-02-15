#!/usr/bin/env python
import numpy as np
import scipy.integrate
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import os, sys

solve_ivp = scipy.integrate.solve_ivp

# Adjust the parent directory as needed (assumes a utils module is available)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from utils import to_pickle, from_pickle

###############################################################################
# 1. ENERGY FUNCTIONS (Vectorized over a batch of states)
###############################################################################
def potential_energy_vec(state, epsilon=1e-8):
    """
    Compute the potential energy for a batch of states.
    Assumes state shape is (batch, nbodies, 7), where for each body:
      index 0: mass
      indexes 1-3: positions
    """
    batch, nbodies, _ = state.shape
    masses = state[:, :, 0:1]    # shape: (batch, nbodies, 1)
    pos = state[:, :, 1:4]       # shape: (batch, nbodies, 3)
    # Compute pairwise displacement vectors: (batch, nbodies, nbodies, 3)
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    # Compute distances; keep an extra dim for broadcasting: (batch, nbodies, nbodies, 1)
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    # Avoid self-interaction by adding a large number on the diagonal
    eye = np.eye(nbodies)[None, :, :, None]
    dist += eye * 1e6
    energy_matrix = masses * masses.transpose(0, 2, 1) / (dist + epsilon)
    tot_energy = -0.5 * energy_matrix.sum(axis=(1, 2))
    return tot_energy.squeeze()

def kinetic_energy_vec(state):
    """
    Compute the kinetic energy for a batch of states.
    Assumes state shape is (batch, nbodies, 7), where for each body:
      index 0: mass
      indexes 4-6: velocities
    """
    ke = 0.5 * state[:, :, 0:1] * np.sum(state[:, :, 4:7]**2, axis=2, keepdims=True)
    return ke.sum(axis=1).squeeze()

def total_energy_vec(state):
    """Return the total energy for each state in the batch."""
    return potential_energy_vec(state) + kinetic_energy_vec(state)

###############################################################################
# 2. ACCELERATION FUNCTIONS
###############################################################################
def get_accelerations(state, epsilon=1e-8):
    """
    Compute the accelerations for a single state.
    This function is used by the ODE integrator and expects a flattened state.
    It reshapes the state to (nbodies, 7) and returns a flattened derivative.
    """
    state = state.reshape(-1, 7)
    nbodies = state.shape[0]
    pos = state[:, 1:4]
    masses = state[:, 0:1]
    # Compute pairwise differences: (nbodies, nbodies, 3)
    diff = pos[None, :, :] - pos[:, None, :]
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    # Prevent self-interaction (diagonal)
    eye = np.eye(nbodies)[:, :, None]
    dist += eye * 1e6
    inv_dist3 = 1.0 / (dist**3 + epsilon)
    acc = np.sum(diff * (masses[None, :, :] * inv_dist3), axis=1)
    return acc

def get_accelerations_vec(state, epsilon=1e-8):
    """
    Compute accelerations in a vectorized way.
    Assumes state has shape (batch, nbodies, 7).
    Returns accelerations with shape (batch, nbodies, 3).
    """
    pos = state[:, :, 1:4]    # (batch, nbodies, 3)
    masses = state[:, :, 0:1] # (batch, nbodies, 1)
    diff = pos[:, None, :, :] - pos[:, :, None, :]  # (batch, nbodies, nbodies, 3)
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    inv_dist3 = 1.0 / (dist**3 + epsilon)
    acc = np.sum(diff * (masses[:, None, :, :] * inv_dist3), axis=2)
    return acc

###############################################################################
# 3. UPDATE FUNCTIONS
###############################################################################
def update(t, state):
    """
    Update function for ODE integration.
    Reshapes the state to (nbodies, 7), computes derivatives, and returns
    a flattened derivative.
    """
    state = state.reshape(-1, 7)
    deriv = np.zeros_like(state)
    deriv[:, 1:4] = state[:, 4:7]         # derivative of position = velocity
    deriv[:, 4:7] = get_accelerations(state)  # derivative of velocity = acceleration
    return deriv.reshape(-1)

def update_vec(state):
    """
    Vectorized update function for a batch of states.
    Assumes state has shape (batch, nbodies, 7) and returns the derivative
    with the same shape.
    """
    deriv = np.zeros_like(state)
    deriv[:, :, 1:4] = state[:, :, 4:7]
    deriv[:, :, 4:7] = get_accelerations_vec(state)
    return deriv

###############################################################################
# 4. ORBIT INTEGRATION AND INITIALIZATION
###############################################################################
def get_orbit(state, update_fn=update, t_points=100, t_span=[0, 2], nbodies=3, **kwargs):
    """
    Integrate an orbit starting from a given initial state.
    Returns the integrated orbit with shape (nbodies, 7, t_points) and integration settings.
    """
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-6

    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(),
                     t_eval=t_eval, **kwargs)
    orbit = path['y'].reshape(nbodies, 7, t_points)
    settings = {'t_eval': t_eval, 't_span': t_span, 'rtol': kwargs['rtol']}
    return orbit, settings

def rotate3d(p, theta, axis):
    """
    Rotate a 3D point p by angle theta about the specified axis.
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(3)
    if axis == 'x':
        R[1:, 1:] = [[c, -s], [s, c]]
    elif axis == 'y':
        R[::2, ::2] = [[c, s], [-s, c]]
    elif axis == 'z':
        R[:2, :2] = [[c, -s], [s, c]]
    return (R @ p.reshape(3, 1)).squeeze()

def random_config(nu=2e-1, min_radius=0.9, max_radius=1.2):
    """
    Generate a random initial configuration for a three-body system.
    Returns a state array with shape (3, 7).
    """
    state = np.zeros((3, 7))
    state[:, 0] = 1  # All bodies have unit mass
    p1 = 2 * np.random.rand(3) - 1
    r = np.random.rand() * (max_radius - min_radius) + min_radius
    p1 *= r / np.linalg.norm(p1)
    p2 = rotate3d(p1, theta=2 * np.pi / 3, axis='z')
    p3 = rotate3d(p2, theta=2 * np.pi / 3, axis='z')
    v1 = rotate3d(p1, theta=np.pi / 2, axis='z') / (r**1.5)
    # Scale v1 to satisfy circular orbit dynamics
    v1 *= np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6)**2))
    v2 = rotate3d(v1, theta=2 * np.pi / 3, axis='z')
    v3 = rotate3d(v2, theta=2 * np.pi / 3, axis='z')
    # Add a small random noise to the velocities
    v1 *= 1 + nu * (2 * np.random.rand(3) - 1)
    v2 *= 1 + nu * (2 * np.random.rand(3) - 1)
    v3 *= 1 + nu * (2 * np.random.rand(3) - 1)
    state[0, 1:4], state[0, 4:7] = p1, v1
    state[1, 1:4], state[1, 4:7] = p2, v2
    state[2, 1:4], state[2, 4:7] = p3, v3
    return state

###############################################################################
# 5. SIMULATION OF ORBITS (Vectorized Post–Processing)
###############################################################################
def simulate_orbit_vectorized(timesteps, nbodies, orbit_noise, min_radius, max_radius, t_span, **kwargs):
    """
    Simulate one orbit and process all time steps in a vectorized manner.
    Returns:
      coords: array of shape (timesteps, nbodies*3) containing positions,
      dcoords: array of shape (timesteps, nbodies*3) containing velocities,
      energy:  array of shape (timesteps,) containing the total energy.
    """
    # Generate a random initial configuration.
    state = random_config(nu=orbit_noise, min_radius=min_radius, max_radius=max_radius)
    
    # Integrate the orbit (resulting shape: (nbodies, 7, timesteps))
    orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
    
    # Rearrange so that time is the first axis: (timesteps, nbodies, 7)
    batch = orbit.transpose(2, 0, 1)
    
    # Extract coordinates (skip mass, i.e. index 0) and flatten each time step:
    coords = batch[:, :, 1:4].transpose(0, 2, 1).reshape(timesteps, -1)
    
    # Compute time derivatives (velocities) in batch.
    dstate = update_vec(batch)
    dcoords = dstate[:, :, 1:4].transpose(0, 2, 1).reshape(timesteps, -1)
    
    # Compute the energy at each time step.
    energy = total_energy_vec(batch)
    
    return coords, dcoords, energy

###############################################################################
# 6. PARALLEL SAMPLING AND DATASET CONSTRUCTION
###############################################################################
def worker_func(args, orbit_kwargs):
    """
    Worker function for multiprocessing.
    Unpacks the arguments and passes additional keyword arguments.
    """
    return simulate_orbit_vectorized(*args, **orbit_kwargs)

def sample_orbits_parallel(timesteps=20, trials=5000, nbodies=3, orbit_noise=2e-1,
                           min_radius=0.9, max_radius=1.2, t_span=[0, 5],
                           nprocs=mp.cpu_count(), **kwargs):
    """
    Run orbit simulations in parallel using multiprocessing.
    Each process simulates one orbit (yielding several time steps/samples).
    """
    # Create a list of argument tuples, one per trial.
    arg_list = [(timesteps, nbodies, orbit_noise, min_radius, max_radius, t_span)
                for _ in range(trials)]
    
    # Prepare a worker function that includes additional kwargs via partial.
    worker = partial(worker_func, orbit_kwargs=kwargs)
    
    pool = mp.Pool(nprocs)
    results = []
    for res in tqdm(pool.imap_unordered(worker, arg_list), total=len(arg_list), desc="Generating orbits"):
        results.append(res)
    pool.close()
    pool.join()
    
    all_x, all_dx, all_e = [], [], []
    for x, dx, e in results:
        all_x.append(x)
        all_dx.append(dx)
        all_e.append(e)
    
    # Concatenate along the time axis.
    data = {
        'coords': np.concatenate(all_x, axis=0),
        'dcoords': np.concatenate(all_dx, axis=0),
        'energy': np.concatenate(all_e, axis=0)
    }
    return data

def make_orbits_dataset(test_split=0.2, **kwargs):
    data = sample_orbits_parallel(**kwargs)
    # Make a train/test split.
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    split_data['meta'] = kwargs
    return split_data

def get_dataset(experiment_name, save_dir, **kwargs):
    """
    Returns an orbital dataset. Constructs the dataset if no saved version is available.
    """
    path = '{}/{}-3Dorbits-dataset.pkl'.format(save_dir, experiment_name)
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except Exception as ex:
        print("Problem loading data from {}: {}. Rebuilding dataset...".format(path, ex))
        data = make_orbits_dataset(**kwargs)
        to_pickle(data, path)
    return data

###############################################################################
# 7. MAIN
###############################################################################
if __name__ == '__main__':
    # Example: adjust timesteps, trials, etc. as needed.
    data = sample_orbits_parallel(timesteps=20, trials=10, nbodies=3, 
                                  orbit_noise=2e-1, min_radius=0.9, max_radius=1.2, 
                                  t_span=[0, 5])
    print("Data shapes:")
    print("coords:", data['coords'].shape)
    print("dcoords:", data['dcoords'].shape)
    print("energy:", data['energy'].shape)
