"""Plots the statistically different connections between the control and depressed populations across the whole brain.

NOTE: Uses existing FunctionalConnectivity object saved in data/two_hemi_fc.pkl. If you want to run this script, you must first run src/dataset/save_functional_connectivity.py to generate this object.

Requirements: conda activate ddc-eda

Usage: python3 plot_brain_matrix.py
"""

import pickle
import sys

sys.path.insert(0, "../../src")

with open("../../data/two_hemi_fc.pkl", "rb") as f:
    fc = pickle.load(f)

fc.plot_significant_connections_matrix(save_as="whole-brain/brain_matrix.png",bonferroni=False)
