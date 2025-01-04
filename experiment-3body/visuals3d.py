import os, sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch, time, sys
import numpy as np
import scipy.integrate, scipy.ndimage
solve_ivp = scipy.integrate.solve_ivp
gaussian_filter = scipy.ndimage.gaussian_filter

EXPERIMENT_DIR = './experiment-3body'
sys.path.append(EXPERIMENT_DIR)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from utils import L2_loss, to_pickle, from_pickle
from data3d import get_dataset, get_orbit, random_config
from data3d import potential_energy, kinetic_energy, total_energy

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
         'fig_dir': './figures/3Dfigures'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


def plot_ground_truth():
    args = ObjectView(get_args())
    np.random.seed(0)
    state = random_config()
    orbit, settings = get_orbit(state, t_points=1000, t_span = [0, 20], rtol = 1e-9) # original t_span = [0,5]

    # print('orbit shape:', orbit.shape)
    # # print('settings shape:', settings.shape)

    # print(orbit)
    # print('\n\n')
    # print(settings)
    
    # draw trajectories
    fig = plt.figure(figsize=[10,4], dpi=100)
    ax = fig.add_subplot(1, 2, 1, projection='3d')  # 3D subplot
    #plt.subplot(1,2,1)
    ax.set_title('Trajectories')
    z_placeholder = np.zeros(1000) # z_coord testing
    cooler_placeholder = np.linspace(-1, 1, 1000)  # Numbers from 1 to 1000
    for i, path in enumerate(orbit):
        plt.plot(path[1], path[2], cooler_placeholder, label='body {} path'.format(i))
    
    ax.axis('equal')
    ax.set_xlabel('$x$') ; ax.set_ylabel('$y$') ; ax.set_zlabel('$z$')
    ax.legend(fontsize=8)

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
    print('Figure saved in:', PARENT_DIR + args.fig_dir, 'as', 'orbits-dataset.{}'.format(FORMAT))


# def load_model(args, baseline=False):
#     output_dim = args.input_dim if baseline else 2
#     nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
#     model = HNN(args.input_dim, differentiable_model=nn_model,
#             field_type=args.field_type, baseline=baseline)
    
#     case = 'baseline' if baseline else 'hnn'
#     path = "{}/{}-orbits-{}.tar".format(args.save_dir, args.name, case)
#     model.load_state_dict(torch.load(path))
#     return model

# args = ObjectView(get_args())
# base_model = load_model(args, baseline=True)
# hnn_model = load_model(args, baseline=False)

def model_update(t, state, model):
    state = state.reshape(-1,5)

    deriv = np.zeros_like(state)
    np_x = state[:,1:] # drop mass
    np_x = np_x.T.flatten()[None, :]
    x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_derivative(x)
    deriv[:,1:] = dx_hat.detach().data.numpy().reshape(4,3).T
    return deriv.reshape(-1)

def what_has_baseline_learned():

    global base_orbit

    # for integrating a vector field parameterized by a NN or HNN

    np.random.seed(0)
    t_points = 2000
    t_span = [0,5]
    state = random_config()

    orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span)

    update_fn = lambda t, y0: model_update(t, y0, base_model)
    base_orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span, update_fn=update_fn)

    lw = 3 #linewidth
    fs=9
    ts=15
    tpad = 7
    ls=12

    fig = plt.figure(figsize=[15,4], dpi=100)
    plt.subplot(1,3,1)
    plt.title('Trajectories', fontsize=ts, pad=tpad)
    colors = ['orange', 'purple', 'blue']
    for i, path in enumerate(orbit):
        plt.plot(path[1], path[2], '-', c=colors[i], label='True path, body {}'.format(i), linewidth=2)
        
    for i, path in enumerate(base_orbit):
        plt.plot(path[1], path[2], '--', c=colors[i], label='NN path, body {}'.format(i), linewidth=2)

    plt.axis('equal')
    plt.xlabel('$x$', fontsize=ls) ; plt.ylabel('$y$', fontsize=ls)
    plt.legend(fontsize=fs)

    plt.subplot(1,3,2)
    real_pe, real_ke, real_etot = potential_energy(orbit), kinetic_energy(orbit), total_energy(orbit)
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

    plt.subplot(1,3,3)
    plt.title('Baseline NN energy', fontsize=ts, pad=tpad)
    plt.xlabel('Time')
    plt.plot(settings['t_eval'], potential_energy(base_orbit), 'g:', label='Potential', linewidth=lw)
    plt.plot(settings['t_eval'], kinetic_energy(base_orbit), 'c-.', label='Kinetic', linewidth=lw)
    plt.plot(settings['t_eval'], total_energy(base_orbit), 'k-', label='Total', linewidth=lw)
    plt.legend(fontsize=fs)
    plt.xlim(*settings['t_span'])
    plt.ylim(ymin, ymax)

    plt.tight_layout() ; plt.show()
    fig.savefig('{}/3body-base-example.{}'.format(args.fig_dir, FORMAT))
    print('Figure saved in:', PARENT_DIR + args.fig_dir, 'as', '3body-base-example.{}'.format(FORMAT))


def what_has_hnn_learned():

    global hnn_orbit

    np.random.seed(0)
    t_points = 2000
    t_span = [0,5]
    state = random_config()

    orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span)

    update_fn = lambda t, y0: model_update(t, y0, hnn_model)
    hnn_orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span, update_fn=update_fn)

    lw = 3 #linewidth
    fs=9
    ts=15
    tpad = 7
    ls=12

    fig = plt.figure(figsize=[15,4], dpi=100)
    plt.subplot(1,3,1)
    plt.title('Trajectories', fontsize=ts, pad=tpad)
    colors = ['orange', 'purple', 'blue']
    for i, path in enumerate(orbit):
        plt.plot(path[1], path[2], '-', c=colors[i], label='True path, body {}'.format(i), linewidth=2)
        
    for i, path in enumerate(hnn_orbit):
        plt.plot(path[1], path[2], '--', c=colors[i], label='NN path, body {}'.format(i), linewidth=2)

    plt.axis('equal')
    plt.xlabel('$x$', fontsize=ls) ; plt.ylabel('$y$', fontsize=ls)
    plt.legend(fontsize=fs)

    plt.subplot(1,3,2)
    real_pe, real_ke, real_etot = potential_energy(orbit), kinetic_energy(orbit), total_energy(orbit)
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

    plt.subplot(1,3,3)
    plt.title('Baseline NN energy', fontsize=ts, pad=tpad)
    plt.xlabel('Time')
    plt.plot(settings['t_eval'], potential_energy(hnn_orbit), 'g:', label='Potential', linewidth=lw)
    plt.plot(settings['t_eval'], kinetic_energy(hnn_orbit), 'c-.', label='Kinetic', linewidth=lw)
    plt.plot(settings['t_eval'], total_energy(hnn_orbit), 'k-', label='Total', linewidth=lw)
    plt.legend(fontsize=fs)
    plt.xlim(*settings['t_span'])
    plt.ylim(ymin, ymax)

    plt.tight_layout() ; plt.show()
    fig.savefig('{}/3body-hnn-example.{}'.format(args.fig_dir, FORMAT))
    print('Figure saved in:', PARENT_DIR + args.fig_dir, 'as', '3body-hnn-example.{}'.format(FORMAT))



def visualize_all_orbits():

    np.random.seed(0)
    t_points = 2000
    t_span = [0,5]
    state = random_config()

    orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span)

    def plot_orbits(fig, k, tail=10000, pfrac=0.05, fs=28, ms=40, lw=3, tpad=15):
        xmin = ymin = np.min([orbit[:,1,:].min(), orbit[:,2,:].min()])
        xmax = ymax = np.max([orbit[:,1,:].max(), orbit[:,2,:].max()])
        pad = (xmax-xmin)*pfrac
        xmin -= pad
        xmax += pad
        ymin -= pad
        ymax += pad

        colors = [(0,0,0), (.6,.6,.6), (.8,.8,.8)]
        t = max(0, k-tail)

        plt.subplot(1,3,1)
        plt.title('Ground truth', fontsize=fs, pad=tpad)
        for i, path in enumerate(orbit):
            plt.plot(path[1,t:k], path[2,t:k], '--', c=colors[i], linewidth=lw)
        for i, path in enumerate(orbit):
            plt.plot(path[1,k], path[2,k], '.', c=colors[i], markersize=ms)
    #     plt.axis('off')
    #     plt.xticks([], []) ; plt.yticks([], [])
        plt.xlim(xmin, xmax) ; plt.ylim(ymin, ymax)
    #     plt.axis('equal')

        colors = [(1,0,0), (1,.6,.6), (1,.8,.8)]
        plt.subplot(1,3,2)
        plt.title('Baseline NN', fontsize=fs, pad=tpad)
        for i, path in enumerate(base_orbit):
            plt.plot(path[1,t:k], path[2,t:k], '--', c=colors[i], linewidth=lw)
        for i, path in enumerate(base_orbit):
            plt.plot(path[1,k], path[2,k], '.', c=colors[i], markersize=ms)
    #     plt.axis('off')
    #     plt.xticks([], []) ; plt.yticks([], [])
        plt.xlim(xmin, xmax) ; plt.ylim(ymin, ymax)
    #     plt.axis('equal')

        colors = [(0,0,1), (.6,.6,1), (.8,.8,1)]
        plt.subplot(1,3,3)
        plt.title('Hamiltonian NN', fontsize=fs, pad=tpad)
        for i, path in enumerate(hnn_orbit):
            plt.plot(path[1,t:k], path[2,t:k], '--', c=colors[i], linewidth=lw)
        for i, path in enumerate(hnn_orbit):
            plt.plot(path[1,k], path[2,k], '.', c=colors[i], markersize=ms)
    #     plt.axis('off')
    #     plt.xticks([], []) ; plt.yticks([], [])
        plt.xlim(xmin, xmax) ; plt.ylim(ymin, ymax)
    #     plt.axis('equal')
        
        plt.tight_layout()

    dpi = 40
    k = 1600
    fig = plt.figure(figsize=[8,2.8], dpi=DPI)
    plot_orbits(fig, k, fs=13, tpad=6, lw=2, ms=30)
    plt.show() ; fig.savefig('{}/3body-compare.{}'.format(args.fig_dir, FORMAT))
    print('Figure saved in:', PARENT_DIR + args.fig_dir, 'as', '3body-compare.{}'.format(FORMAT))


def visualize_all_energies():

    np.random.seed(0)
    t_points = 2000
    t_span = [0,5]
    state = random_config()

    orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span)

    tail=10000; pfrac=0.05
    fs=13; tpad=6; lw=2; ms=30

    k = 1600
    fig = plt.figure(figsize=[8,2.8], dpi=DPI)

    tstart = max(0, k-tail)

    real_pe = potential_energy(orbit[...,tstart:k])
    real_ke = kinetic_energy(orbit[...,tstart:k])
    real_etot = total_energy(orbit[...,tstart:k])
    ymin = np.min([real_pe.min(), real_ke.min(), real_etot.min()])
    ymax = np.max([real_pe.max(), real_ke.max(), real_etot.max()])
    pad = (ymax-ymin)*pfrac
    ymin -= pad
    ymax += pad

    t = settings['t_eval'][tstart:k]

    plt.subplot(1,3,1)
    plt.title('Ground truth', fontsize=fs, pad=tpad)
    plt.plot(t, real_ke,'k--', linewidth=lw, label='Kinetic')
    plt.plot(t, real_pe,'k:', linewidth=lw, label='Potential')
    plt.plot(t, real_etot,'k-', linewidth=lw, label='Total')
    # plt.xlabel('Time', fontsize=12) #; plt.ylabel("Energy")
    plt.ylim(ymin, ymax)
    plt.legend(fontsize=7, loc='center right')

    plt.subplot(1,3,2)
    plt.title('Baseline NN', fontsize=fs, pad=tpad)
    plt.plot(t, kinetic_energy(base_orbit[...,tstart:k]), 'r--', linewidth=lw, label='Kinetic')
    plt.plot(t, potential_energy(base_orbit[...,tstart:k]), 'r:', linewidth=lw, label='Potential')
    plt.plot(t, total_energy(base_orbit[...,tstart:k]), 'r-', linewidth=lw, label='Total')
    plt.ylim(ymin, ymax)
    # plt.xlabel('Time') ; plt.ylabel("Energy")
    plt.legend(fontsize=7, loc='center right')

    plt.subplot(1,3,3)
    plt.title('Hamiltonian NN', fontsize=fs, pad=tpad)
    plt.plot(t, kinetic_energy(hnn_orbit[...,tstart:k]),'b--', linewidth=lw, label='Kinetic')
    plt.plot(t, potential_energy(hnn_orbit[...,tstart:k]), 'b:', linewidth=lw, label='Potential')
    plt.plot(t, total_energy(hnn_orbit[...,tstart:k]), 'b-', linewidth=lw, label='Total')
    plt.ylim(ymin, ymax)
    # plt.xlabel('Time') ; plt.ylabel("Energy")
    plt.legend(fontsize=7, loc='center right')

    plt.tight_layout() ; fig.savefig('{}/3body-energy-compare.{}'.format(args.fig_dir, FORMAT))
    print('Figure saved in:', PARENT_DIR + args.fig_dir, 'as', '3body-base-example.{}'.format(FORMAT))


def visualize_energy_conservation():

    t_points = 500
    t_span = [0,3]
    trials = 15

    true_energies, base_energies, hnn_energies = [], [], []
    for trial_ix in range(trials):
        
        np.random.seed(trial_ix)
        state = random_config()
        
        # true trajectory -> energy
        orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span)
        true_energies.append(total_energy(orbit))
        
        # baseline NN trajectory -> energy
        update_fn = lambda t, y0: model_update(t, y0, base_model)
        base_orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span, update_fn=update_fn)
        base_energies.append(total_energy(base_orbit))
        
        # hamiltonian NN trajectory -> energy
        update_fn = lambda t, y0: model_update(t, y0, hnn_model)
        hnn_orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span, update_fn=update_fn)
        hnn_energies.append(total_energy(hnn_orbit))
        
    true_energies = np.stack(true_energies)
    base_energies = np.stack(base_energies)
    hnn_energies = np.stack(hnn_energies)

    rows, cols = 3, 4
    N = rows*cols
    s = 4
    fig = plt.figure(figsize=[cols*s, .9*rows*s], dpi=DPI)
    for i in range(N):
        
        plt.subplot(rows, cols,i+1)
        t_domain = np.linspace(t_span[0], t_span[1], t_points)
        plt.title('Random seed={}'.format(i))
        plt.plot(t_domain, true_energies[i], 'k-', label='Ground truth')
        plt.plot(t_domain, base_energies[i], 'r-', label='Baseline NN')
        plt.plot(t_domain, hnn_energies[i], 'b-', label='Hamiltonian NN')
        plt.xlabel('Time')
        plt.ylabel('Total energy')
        plt.legend()
        
    plt.tight_layout() ; plt.show()
    fig.savefig('{}/3body-total-energy.{}'.format(args.fig_dir, FORMAT))
    print('Figure saved in:', PARENT_DIR + args.fig_dir, 'as', '3body-total=energy.{}'.format(FORMAT))


def plot_training_curves():

    base_stats = from_pickle('{}/3body-orbits-baseline.pkl'.format(EXPERIMENT_DIR))
    hnn_stats = from_pickle('{}/3body-orbits-hnn.pkl'.format(EXPERIMENT_DIR))

    fig = plt.figure(figsize=[5,3], dpi=DPI)

    fw = 200 # gaussian filter width
    plt.plot(gaussian_filter(hnn_stats['test_loss'], fw), 'b-', label='hnn-test')
    plt.plot(gaussian_filter(hnn_stats['train_loss'], fw), 'b--', label='hnn-train')

    plt.plot(gaussian_filter(base_stats['test_loss'], fw), 'r-', label='base-test')
    plt.plot(gaussian_filter(base_stats['train_loss'], fw), 'r--', label='base-train')

    plt.title('3-body training curves')
    plt.xlabel('Gradient step') ; plt.ylabel('Loss')

    plt.legend()
    plt.yscale('log')
    plt.tight_layout() ; plt.show()
    fig.savefig('{}/3body-train-curves-long.{}'.format(args.fig_dir, FORMAT))
    print('Figure saved in:', PARENT_DIR + args.fig_dir, 'as', '3body-train-curves-long.{}'.format(FORMAT))

# while True:
#     # Display options
#     print("\nPlease choose an option:")
#     print("1: plot_ground_truth")
#     print("2: load_models")
#     print("3: what_has_baseline_learned")
#     print("4: what_has_hnn_learned")
#     print("5: visualize_all_orbits")
#     print("6: visualize_all_energies")
#     print("7: visualize_energy_conservation")
#     print("8: plot_training_curves")
    
#     print("-1: Exit")
    
#     # Get user input
#     user_choice = input("Enter the number of your choice: ")

#     # Check if user wants to exit
#     if user_choice == "-1":
#         print("Exiting the program.")
#         break

#     # Call corresponding function based on input
#     if user_choice == "1":
#         plot_ground_truth()
#     elif user_choice == "2":
#         load_models()
#     elif user_choice == "3":
#         what_has_baseline_learned()
#     elif user_choice == "4":
#         what_has_hnn_learned()
#     elif user_choice == "5":
#         visualize_all_orbits()
#     elif user_choice == "6":
#         visualize_all_energies()
#     elif user_choice == "7":
#         visualize_energy_conservation()
#     elif user_choice == "8":
#         plot_training_curves()
#     else:
#         print("Invalid choice, please enter a number from the list.")

# plot_ground_truth()
# what_has_baseline_learned()
# what_has_hnn_learned()