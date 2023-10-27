# TODO: debug this script
import pickle
import sys

sys.path.insert(0, "../../src")

with open("../../data/two_hemi_fc.pkl", "rb") as f:
    fc = pickle.load(f)

fc.plot_network_connectivity_graph_diff("CEN", ev=400, save_as="cen/cen_graph.png")
fc.plot_network_connectivity_graph_diff("DMN", ev=400, save_as="dmn/dmn_graph.png")
fc.plot_network_connectivity_graph_diff("SN", ev=400, save_as="sn/sn_graph.png")
