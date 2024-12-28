import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils import to_pickle
from data import make_orbits_dataset

path = '{}/{}-orbits-dataset.pkl'.format(THIS_DIR, '3body')
data = make_orbits_dataset(verbose=True)
print('Dataset created successfully')
to_pickle(data, path)
print('...saved in,' path)