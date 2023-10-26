import pandas as pd
import pickle
import sys
sys.path.insert(0, '../src')
from FunctionalConnectivity import FunctionalConnectivity

# Load data
labels = pd.read_csv('/cnl/abcd/data/labels/baseline_depr.csv',header=None, index_col=0)
DDC_path = "/cnl/abcd/data/imaging/fmri/rsfmri/interim/DDC/baseline_depr/"
fig_dir = '/home/mwagner/projects/DDC/figures/'

# Define functional connectivity object
fc = FunctionalConnectivity(
    labels,
    n_roi=98,
    thrs=0.1,
    weights_file_name="subc_DDC*.csv",
    DDC_path=DDC_path,
    fig_dir=fig_dir
    )

# Save object for fast loading
pickle.dump(
    fc,
    open("/home/mwagner/projects/DDC/data/two_hemi_fc.pkl", "wb")
    )