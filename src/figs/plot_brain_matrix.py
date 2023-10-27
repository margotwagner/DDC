import pickle
import sys

sys.path.insert(0, "../../src")

with open("../../data/two_hemi_fc.pkl", "rb") as f:
    fc = pickle.load(f)

fc.plot_significant_connections_matrix(save_as="whole-brain/brain_matrix.png")
