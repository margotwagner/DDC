import pickle
import sys

sys.path.insert(0, "../../src")

with open("../../data/two_hemi_fc.pkl", "rb") as f:
    fc = pickle.load(f)

fc.plot_network_connectivity_graph_diff("CEN", save_as="cen/cen_graph.png")
fc.plot_network_connectivity_graph_diff("DMN", save_as="dmn/dmn_graph.png")
fc.plot_network_connectivity_graph_diff("SN", save_as="sn/sn_graph.png")
