# prints dataset and gives info regarding each key in dictionary

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils import from_pickle

path = '{}/{}-3Dorbits-dataset.pkl'.format(THIS_DIR, '3body')

data = from_pickle(path)
print(data)