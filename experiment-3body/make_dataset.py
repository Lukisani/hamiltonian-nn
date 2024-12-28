import os, to_pickle
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


from data import make_orbits_dataset

path = '{}/{}-orbits-dataset.pkl'.format(THIS_DIR, '3body')
data = make_orbits_dataset(verbose=True)
to_pickle(data, path)