# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy
solve_ivp = scipy.integrate.solve_ivp

import os, sys
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from utils import to_pickle, from_pickle

##### ENERGY #####
def potential_energy(state):
    '''U=\sum_i,j>i G m_i m_j / r_ij'''
    tot_energy = np.zeros((1,1,state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i+1,state.shape[0]):
            r_ij = ((state[i:i+1,1:4] - state[j:j+1,1:4])**2).sum(1, keepdims=True)**.5
            m_i = state[i:i+1,0:1]
            m_j = state[j:j+1,0:1]
            tot_energy += m_i * m_j / r_ij
    U = -tot_energy.sum(0).squeeze()
    return U

def kinetic_energy(state):
    '''T=\sum_i .5*m*v^2'''
    energies = .5 * state[:,0:1] * (state[:,4:7]**2).sum(1, keepdims=True)
    T = energies.sum(0).squeeze()
    return T

def total_energy(state):
    return potential_energy(state) + kinetic_energy(state)


##### DYNAMICS #####
def get_accelerations(state, epsilon=0):
    # shape of state is [bodies x properties]
    net_accs = [] # [nbodies x 2]
    for i in range(state.shape[0]): # number of bodies
        other_bodies = np.concatenate([state[:i, :], state[i+1:, :]], axis=0)
        displacements = other_bodies[:, 1:4] - state[i, 1:4] # indexes 1:3 -> pxs, pys, pyz
        distances = (displacements**2).sum(1, keepdims=True)**0.5
        masses = other_bodies[:, 0:1] # index 0 -> mass
        pointwise_accs = masses * displacements / (distances**3 + epsilon) # G=1
        net_acc = pointwise_accs.sum(0, keepdims=True)
        net_accs.append(net_acc)
    net_accs = np.concatenate(net_accs, axis=0)
    return net_accs
  
def update(t, state):
    state = state.reshape(-1,7) # [bodies, properties]
    deriv = np.zeros_like(state)
    deriv[:,1:4] = state[:,4:7] # dx, dy, dz = vx, vy, dz
    deriv[:,4:7] = get_accelerations(state)
    return deriv.reshape(-1)


##### INTEGRATION SETTINGS #####
def get_orbit(state, update_fn=update, t_points=100, t_span=[0,2], nbodies=3, **kwargs):
    if not 'rtol' in kwargs.keys():
        kwargs['rtol'] = 1e-6 # was -9 before...

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
    
    p1 *= r / np.sqrt(np.sum((p1**2)))
    p2 = rotate3d(p1, theta=2*np.pi/3, axis='z')
    p3 = rotate3d(p2, theta=2*np.pi/3, axis='z')

    v1 = rotate3d(p1, theta=np.pi/2, axis='z') / r**1.5
    v1 *= np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))  # Circular orbit scaling
    v2 = rotate3d(v1, theta=2*np.pi/3, axis='z')
    v3 = rotate3d(v2, theta=2*np.pi/3, axis='z')

    v1 *= 1 + nu * (2*np.random.rand(3) - 1)
    v2 *= 1 + nu * (2*np.random.rand(3) - 1)
    v3 *= 1 + nu * (2*np.random.rand(3) - 1)

    state[0, 1:4], state[0, 4:7] = p1, v1
    state[1, 1:4], state[1, 4:7] = p2, v2
    state[2, 1:4], state[2, 4:7] = p3, v3
    return state


##### INTEGRATE AN ORBIT OR TWO #####
def sample_orbits(timesteps=10, trials=500, nbodies=3, orbit_noise=2e-1,
                  min_radius=0.9, max_radius=1.2, t_span=[0, 5], verbose=False, **kwargs): #trials were at 5000 before and timesteps at 20 - reduced to make dataset generation faster
    
    orbit_settings = locals()
    if verbose:
        print("Making a dataset of near-circular 3-body orbits:")
    
    x, dx, e = [], [], []
    N = timesteps*trials
    while len(x) < N:
        if len(x) % 100 == 0:
            print('len(x) =', len(x)) # for debugging

        state = random_config(nu=orbit_noise, min_radius=min_radius, max_radius=max_radius)
        orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
        batch = orbit.transpose(2,0,1).reshape(-1,nbodies*7)

        for state in batch:
            dstate = update(None, state)
            
            # reshape from [nbodies, state] where state=[m, qx, qy, qz, px, py, pz]
            # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
            coords = state.reshape(nbodies,7).T[1:].flatten()
            dcoords = dstate.reshape(nbodies,7).T[1:].flatten()
            x.append(coords)
            dx.append(dcoords)

            shaped_state = state.copy().reshape(nbodies,7,1)
            e.append(total_energy(shaped_state))
    

    data = {'coords': np.stack(x)[:N],
            'dcoords': np.stack(dx)[:N],
            'energy': np.stack(e)[:N] }
    return data, orbit_settings


##### MAKE A DATASET #####
def make_orbits_dataset(test_split=0.2, **kwargs):
    data, orbit_settings = sample_orbits(**kwargs)
    
    # make a train/test split
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data

    data['meta'] = orbit_settings
    return data


##### LOAD OR SAVE THE DATASET #####
def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''

    path = '{}/{}-3Dorbits-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        data = make_orbits_dataset(**kwargs)
        print(data)
        to_pickle(data, path)

    return data