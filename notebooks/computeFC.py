import os
import time
import numpy as np
import glob
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/acamassa/ABCD/DDC/src/py_DDC/')
from FC_functions import *
# Define your functions here, e.g., compute_ddc and compute_fddc if they are not imported

def main():

    # get the time series data to compute DDC
    files_path = input("Enter the time series data path (e.g., /nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/baseline/): ")
    # files_path="/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/baseline/"
    files = glob.glob(f"{files_path}/**/filt_fMRI_segmented**.csv", recursive=True)

    qc = []
    error = []
    ddc_dir = input("Where to save the FC? (e.g., /nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/DDC/baseline/): ")
    # ddc_dir = '/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/DDC/baseline/'

    i = 0
    start = time.time()

    for f in files:
        print(i)
        i += 1
        ts = np.loadtxt(f, delimiter=",", dtype=float)  # load time series data
        expected_index_row = np.arange(0, np.shape(ts)[1])

        # Check if the first row matches the expected index row
        if np.array_equal(ts[0], expected_index_row):
            # Remove the first row
            ts = ts[1:]
        try:
            save_path = os.path.join(ddc_dir, f.split("/")[0], "single_sessions", "subc_DDC2_" + f[-10:])
            if not os.path.exists(save_path):  # check if DDC already exists
                save_dir = os.path.join(ddc_dir, f.split("/")[0], "single_sessions")
                if not os.path.exists(save_dir):  # make saving directory if needed
                    print("making dir")
                    os.makedirs(save_dir)

                print(f.split("/")[0])
                # compute covariance, DDC, and regularized DDC
                Cov, DDC2, Reg_DDC2, nl_DDC2, DDCc, Reg_DDCc, nl_DDCc, qc_flag = compute_ddc(ts, 0.8)
                qc.append([f, qc_flag])
                pd.DataFrame(Cov).to_csv(
                    os.path.join(save_dir, "subc_Cov_" + f[-10:]),
                    header=None,
                    index=None,
                )
                pd.DataFrame(DDC2).to_csv(
                    os.path.join(save_dir, "subc_DDC2_" + f[-10:]),
                    header=None,
                    index=None,
                )
                pd.DataFrame(Reg_DDC2).to_csv(
                    os.path.join(save_dir, "subc_Reg_DDC2_" + f[-10:]),
                    header=None,
                    index=None,
                )
                pd.DataFrame(nl_DDC2).to_csv(
                    os.path.join(save_dir, "subc_nlDDC2_" + f[-10:]),
                    header=None,
                    index=None,
                )

                pd.DataFrame(DDCc).to_csv(
                    os.path.join(save_dir, "subc_DDCc_" + f[-10:]),
                    header=None,
                    index=None,
                )
                pd.DataFrame(Reg_DDCc).to_csv(
                    os.path.join(save_dir, "subc_Reg_DDCc_" + f[-10:]),
                    header=None,
                    index=None,
                )
                pd.DataFrame(nl_DDCc).to_csv(
                    os.path.join(save_dir, "subc_nlDDCc_" + f[-10:]),
                    header=None,
                    index=None,
                )


                # compute fractional DDC and regularized fractional DDC
                FDDC, Reg_FDDC = compute_fddc(ts, 0.8, 0.5)
                pd.DataFrame(FDDC).to_csv(
                    os.path.join(save_dir, "subc_FDDC_" + f[-10:]),
                    header=None,
                    index=None,
                )
                pd.DataFrame(Reg_FDDC).to_csv(
                    os.path.join(save_dir, "subc_Reg_FDDC_" + f[-10:]),
                    header=None,
                    index=None,
                )

        except Exception as e:
            print('error:', e)
            error.append(f)

    end = time.time()
    print("Execution time:", end - start)


if __name__ == "__main__":
    # Make sure to define `files` and any missing functions, like `compute_ddc` and `compute_fddc`, before running
    main()
