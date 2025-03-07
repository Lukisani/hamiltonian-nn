# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy.integrate
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

solve_ivp = scipy.integrate.solve_ivp

import os, sys
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from utils import to_pickle, from_pickle

##### ENERGY #####
def potential_energy(state):
    '''U=\sum_i,j>i G m_i m_j / r_ij'''
    tot_energy = np.zeros((1, 1, state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i + 1, state.shape[0]):
            r_ij = ((state[i:i+1, 1:4] - state[j:j+1, 1:4])**2).sum(1, keepdims=True)**0.5
            m_i = state[i:i+1, 0:1]
            m_j = state[j:j+1, 0:1]
            tot_energy += m_i * m_j / r_ij
    U = -tot_energy.sum(0).squeeze()
    return U

def kinetic_energy(state):
    '''T=\sum_i .5*m*v^2'''
    energies = 0.5 * state[:, 0:1] * (state[:, 4:7]**2).sum(1, keepdims=True)
    T = energies.sum(0).squeeze()
    return T

def total_energy(state):
    return potential_energy(state) + kinetic_energy(state)

##### DYNAMICS #####
def get_accelerations(state, epsilon=1e-3):
    # shape of state is [bodies x properties]
    net_accs = []  # [nbodies x 2]
    for i in range(state.shape[0]):  # number of bodies
        other_bodies = np.concatenate([state[:i, :], state[i+1:, :]], axis=0)
        displacements = other_bodies[:, 1:4] - state[i, 1:4]  # indexes 1:3 -> positions
        distances = (displacements**2).sum(1, keepdims=True)**0.5
        masses = other_bodies[:, 0:1]  # index 0 -> mass
        pointwise_accs = masses * displacements / (distances**3 + epsilon)  # G=1
        net_acc = pointwise_accs.sum(0, keepdims=True)
        net_accs.append(net_acc)
    net_accs = np.concatenate(net_accs, axis=0)
    return net_accs
  
def update(t, state):
    state = state.reshape(-1, 7)  # [bodies, properties]
    deriv = np.zeros_like(state)
    deriv[:, 1:4] = state[:, 4:7]  # dx, dy, dz = vx, vy, vz
    deriv[:, 4:7] = get_accelerations(state)
    return deriv.reshape(-1)

##### INTEGRATION SETTINGS #####
def get_orbit(state, update_fn=update, t_points=100, t_span=[0, 2], nbodies=3, **kwargs):
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-6  # was -9 before...

    orbit_settings = locals()

    nbodies = state.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    orbit_settings['t_eval'] = t_eval

    path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(),
                     t_eval=t_eval, **kwargs)
    orbit = path['y'].reshape(nbodies, 7, t_points)
    return orbit, orbit_settings

##### INITIALIZE THE THREE BODIES #####
def rotate3d(p, theta, axis):
    # General 3D rotation matrix for a given axis
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
    state = np.zeros((3, 7))
    state[:, 0] = 1  # Masses
    p1 = 2 * np.random.rand(3) - 1
    r = np.random.rand() * (max_radius - min_radius) + min_radius
    
    p1 *= r / np.sqrt(np.sum(p1**2))
    p2 = rotate3d(p1, theta=2 * np.pi / 3, axis='z')
    p3 = rotate3d(p2, theta=2 * np.pi / 3, axis='z')

    v1 = rotate3d(p1, theta=np.pi / 2, axis='z') / r**1.5
    v1 *= np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6)**2))  # Circular orbit scaling
    v2 = rotate3d(v1, theta=2 * np.pi / 3, axis='z')
    v3 = rotate3d(v2, theta=2 * np.pi / 3, axis='z')

    v1 *= 1 + nu * (2 * np.random.rand(3) - 1)
    v2 *= 1 + nu * (2 * np.random.rand(3) - 1)
    v3 *= 1 + nu * (2 * np.random.rand(3) - 1)

    state[0, 1:4], state[0, 4:7] = p1, v1
    state[1, 1:4], state[1, 4:7] = p2, v2
    state[2, 1:4], state[2, 4:7] = p3, v3
    return state

##### INTEGRATE AN ORBIT OR TWO #####
def simulate_orbit(timesteps, nbodies, orbit_noise, min_radius, max_radius, t_span, **kwargs):
    """
    Simulate a single orbit and return the list of sample points (coords, dcoords, energy).
    """
    x, dx, e = [], [], []
    
    # Generate a random initial configuration
    state = random_config(nu=orbit_noise, min_radius=min_radius, max_radius=max_radius)
    
    # Integrate the orbit
    orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
    
    # Reshape the orbit to iterate over time steps (batch: [timesteps, nbodies, 7])
    batch = orbit.transpose(2, 0, 1).reshape(-1, nbodies * 7)
    
    for state_flat in batch:
        dstate_flat = update(None, state_flat)
        
        # Convert the flat state into a shape [nbodies, 7]
        state_reshaped = state_flat.reshape(nbodies, 7)
        dstate_reshaped = dstate_flat.reshape(nbodies, 7)
        
        # Extract canonical coordinates (skip mass, index 0) and flatten them
        coords = state_reshaped.T[1:].flatten()
        dcoords = dstate_reshaped.T[1:].flatten()
        x.append(coords)
        dx.append(dcoords)
        
        # Compute the energy from the full state
        shaped_state = state_flat.copy().reshape(nbodies, 7, 1)
        e.append(total_energy(shaped_state))
        
    return x, dx, e

# --- Worker function for multiprocessing with extra kwargs ---
def worker_func(args, orbit_kwargs):
    """
    Unpacks args and passes additional keyword arguments.
    This function is defined at the top level so it can be pickled.
    """
    np.random.seed(mp.current_process().pid)
    return simulate_orbit(*args, **orbit_kwargs)

##### PARALLEL ORBIT SAMPLING WITH PROGRESS BAR #####
def sample_orbits_parallel(timesteps=20, trials=5000, nbodies=3, orbit_noise=2e-1,
                           min_radius=0.9, max_radius=1.2, t_span=[0, 5],
                           nprocs=2*mp.cpu_count(), **kwargs): # trials were at 5000 originally
    """
    Run orbit simulations in parallel using multiprocessing.
    Each process simulates one orbit (which gives several time steps/samples).
    A tqdm progress bar is displayed to show progress.
    """
    # Create a list of argument tuples, one per trial
    arg_list = [(timesteps, nbodies, orbit_noise, min_radius, max_radius, t_span)
                for _ in range(trials)]
    
    # Prepare a worker function that includes additional kwargs via partial
    worker = partial(worker_func, orbit_kwargs=kwargs)
    
    pool = mp.Pool(nprocs)
    results = []
    # Use imap_unordered wrapped with tqdm for a progress bar
    for res in tqdm(pool.imap_unordered(worker, arg_list), total=len(arg_list), desc="Generating orbits"):
        results.append(res)
    
    pool.close()
    pool.join()
    
    # Collect results from each orbit simulation
    all_x, all_dx, all_e = [], [], []
    for x, dx, e in results:
        all_x.extend(x)
        all_dx.extend(dx)
        all_e.extend(e)
        
    # Convert lists to numpy arrays
    data = {
        'coords': np.stack(all_x),
        'dcoords': np.stack(all_dx),
        'energy': np.stack(all_e)
    }
    return data

##### MAKE A DATASET #####
def make_orbits_dataset(test_split=0.2, **kwargs):
    data = sample_orbits_parallel(**kwargs)
    
    # Make a train/test split
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data

    data['meta'] = kwargs  # You can add additional metadata as needed
    return data

##### LOAD OR SAVE THE DATASET #####
def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''

    path = '{}/{}-3Dorbits-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except Exception as ex:
        print("Had a problem loading data from {}: {}. Rebuilding dataset...".format(path, ex))
        data = make_orbits_dataset(**kwargs)
        to_pickle(data, path)

    return data

# For multiprocessing safety on Windows, include this guard
if __name__ == '__main__':
    # Example: adjust timesteps, trials, etc. as needed.
    data = sample_orbits_parallel(timesteps=20, trials=10, nbodies=3, 
                                  orbit_noise=2e-1, min_radius=0.9, max_radius=1.2, 
                                  t_span=[0, 5])
    print("Data shapes:")
    print("coords:", data['coords'].shape)
    print("dcoords:", data['dcoords'].shape)
    print("energy:", data['energy'].shape)
