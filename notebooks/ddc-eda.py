import sys

sys.path.insert(0, "../src")
from FunctionalConnectivity import FunctionalConnectivity
import pandas as pd


labels_dir = "/cnl/abcd/data/labels/"
labels = pd.read_csv(labels_dir + "baseline_depr.csv", header=None, index_col=0)

two_hemi_fc = FunctionalConnectivity(
    labels, n_roi=68, thrs=0.01, weights_file_name="Reg_DDC2H_*.csv"
)


# two_hemi_fc.plot_significant_connections_matrix()

two_hemi_fc.plot_network_heatmap("DMN", "RdBu", "Greys")

# two_hemi_fc.plot_network_heatmap("CEN", "RdBu", "Greys")

# two_hemi_fc.plot_network_heatmap("SN", "RdBu", "Greys")
