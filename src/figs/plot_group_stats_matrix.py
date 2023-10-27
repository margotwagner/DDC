import pickle
import sys

sys.path.insert(0, "../../src")

with open("../../data/two_hemi_fc.pkl", "rb") as f:
    fc = pickle.load(f)

fc.plot_means_std_matrices(save_as="whole-brain/means_std_matrices.png")
