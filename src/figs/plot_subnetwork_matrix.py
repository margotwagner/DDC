"""This script plots the subnetwork matrices for the CEN, DMN, and SN for control and depressed subjects as well as the connections that are statistically different.

Usage:
    conda activate ddc-eda
    python3 plot_subnetwork_matrix.py
"""
import pickle
import sys

sys.path.insert(0, "../../src")

with open("../../data/two_hemi_fc.pkl", "rb") as f:
    fc = pickle.load(f)

fc.plot_network_heatmap("CEN", save_as="cen/cen_matrix.png")
fc.plot_network_heatmap("DMN", save_as="dmn/dmn_matrix.png")
fc.plot_network_heatmap("SN", save_as="sn/sn_matrix.png")
