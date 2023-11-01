"""Builds dataset for running classification on subnetworks.

Usage: conda activate ddc-eda
python3 ./build_subnetwork_dataset.py
"""


import sys

sys.path.append("../../src")
from FunctionalConnectivity import FunctionalConnectivity
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main(subnetwork):
    DDC_path = "/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/DDC/baseline_depr/"
    fig_dir = "../figures/"

    # Get labels
    labels = pd.read_csv(
        "/cnl/abcd/data/labels/baseline_depr.csv", header=None, index_col=0
    )

    # Two hemisphere FC
    fc = FunctionalConnectivity(
        labels,
        n_roi=98,
        thrs=0.1,
        weights_file_name="subc_DDC*.csv",
        DDC_path=DDC_path,
        fig_dir=fig_dir,
    )

    # Build dataset and standard scale
    ctrl = fc.get_flat_network_ddc(subnetwork, "control")
    depr = fc.get_flat_network_ddc(subnetwork, "depr")
    data = np.concatenate((ctrl, depr), axis=0)
    data = StandardScaler().fit_transform(data)
    print(f"Data shape: {data.shape}")

    # Organized dataset
    data = pd.DataFrame(data, index=fc.control_subj_ids + fc.depress_subj_ids)

    data.to_csv(f"../../data/processed/ddc-{subnetwork.lower()}.csv")

    print(
        f"Standard scaled dataset for {subnetwork} saved to data/processed/ddc-{subnetwork.lower()}.csv"
    )


if __name__ == "__main__":
    main("SN")
