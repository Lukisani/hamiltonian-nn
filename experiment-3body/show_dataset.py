# prints dataset and gives info regarding each key in dictionary

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils import from_pickle

path = '{}/{}-orbits-dataset.pkl'.format(THIS_DIR, '3body')

data = from_pickle(path)
print(data)
print()
print('coords shape:', data['coords'].shape)
print('test_coords:', data['test_coords'].shape)
print('dcoords shape:', data['dcoords'].shape)
print('test_dcoords shape:', data['test_dcoords'].shape)
print('energy shape:', data['energy'].shape)
print('test_energy shape:', data['test_energy'].shape)
print('meta:')
for key, value in data['meta'].items():
    print('\t' + key + ':', value)