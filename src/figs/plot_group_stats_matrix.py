"""Plots the group mean and std matrices for the whole brain FC data for both control and depressed populations.

NOTE: Uses existing FunctionalConnectivity object saved in data/two_hemi_fc.pkl. If you want to run this script, you must first run src/dataset/save_functional_connectivity.py to generate this object.

Requirements: conda activate ddc-eda

Usage: python3 plot_group_stats_matrix.py
"""

import pickle
import sys
sys.path.insert(0, "../../src")

SAVE_LOC = "whole-brain/means_std_matrices.png"

with open("../../data/two_hemi_fc.pkl", "rb") as f:
    fc = pickle.load(f)

fc.plot_means_std_matrices(cmap='Greens', save_as=SAVE_LOC)

print(f"Saved matrices of whole brain avg and std in each population to {SAVE_LOC}")