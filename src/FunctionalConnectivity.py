# Class for DDC dataset
# TODO: generalize functions so they can take in depressed *or* control
# TODO: better way to handle network name -- assign globally?
# TODO: comment!

import numpy as np
import glob
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import mannwhitneyu
from nilearn import datasets, plotting
from scipy.stats import mannwhitneyu
import seaborn as sns
from scipy.stats import ttest_ind


class FunctionalConnectivity:
    def __init__(
        self,
        labels,
        n_roi,  # TODO: specify from data?
        thrs,  # binarizatin threshold
        weights_file_name,
        DDC_path,
        fig_dir,
    ):
        self.labels = labels
        self.n_roi = n_roi
        self.thrs = thrs
        self.weights_file_name = weights_file_name
        self.DDC_path = DDC_path
        self.fig_dir = fig_dir

        # TODO: specify from data rather than hardcoding
        self.n1 = 5000
        self.n2 = 5000

        self.subc = [
            "l-CWM",
            "l-LV",
            "l-ILV",
            "l-CBWM",
            "l-CBC",
            "l-TH",
            "l-CAU",
            "l-PUT",
            "l-GP",
            "3V",
            "4V",
            "BSM",
            "l-HPC",
            "l-Amy",
            "CSF",
            "l-NAcc",
            "l-vDC",
            "r-CWM",
            "r-LV",
            "r-ILV",
            "r-CBWM",
            "r-CBC",
            "r-TH",
            "r-CAU",
            "r-PUT",
            "r-GP",
            "r-HPC",
            "r-Amy",
            "r-NAcc",
            "r-vDC",
        ]

        self.left = [
            "l-bsts",
            "l-CACg",
            "l-CMF",
            "l-Cu",
            "l-En",
            "l-Fu",
            "l-IP",
            "l-IT",
            "l-IstCg",
            "l-LO",
            "l-LOrF",
            "l-Lg",
            "l-MOrF",
            "l-MT",
            "l-PaH",
            "l-PaC",
            "l-Op",
            "l-Or",
            "l-Tr",
            "l-PerCa",
            "l-PoC",
            "l-PoCg",
            "l-PreC",
            "l-PreCu",
            "l-RoACg",
            "l-RoMF",
            "l-SF",
            "l-SP",
            "l-ST",
            "l-SM",
            "l-FPol",
            "l-TPol",
            "l-TrT",
            "l-Ins",
        ]

        self.right = ["r" + label[1:] for label in self.left]

        # Concatenate the two lists
        self.all_ROIs = self.subc + self.left + self.right

        self.DMN_indices = [12, 21, 6, 23, 31, 46, 55, 40, 57, 65]
        if self.weights_file_name.startswith("subc_"):
            y = 30
            self.DMN_indices = list(map(y.__add__, self.DMN_indices))
        self.DMN_labels = [self.all_ROIs[i] for i in self.DMN_indices]

        self.CEN_indices = [6, 12, 25, 26, 27, 40, 46, 59, 60, 61]
        if self.weights_file_name.startswith("subc_"):
            y = 30
            self.CEN_indices = list(map(y.__add__, self.CEN_indices))

        self.CEN_labels = [self.all_ROIs[i] for i in self.CEN_indices]

        self.SN_indices = [33, 24, 31, 67, 58, 65]
        if self.weights_file_name.startswith("subc_"):
            y = 30
            self.SN_indices = list(map(y.__add__, self.SN_indices))
        self.SN_labels = [self.all_ROIs[i] for i in self.SN_indices]

        if self.weights_file_name.startswith("subc_"):
            self.positions = pd.read_csv(
                "/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/NO_baseline/downloads/sub-NDARINV04GAB2AA/ROIs_centroid_coordinates.csv"
            )
        else:
            self.positions = pd.read_csv(
                "/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/NO_baseline/downloads/sub-NDARINV04GAB2AA/ROIs_centroid_coordinates.csv"
            )[30:]

        # builds the dataset and assign the output to the variables
        (
            self.no_weights,
            self.missing_rois,
            self.depr_files,
            self.ctrl_files,
            self.control_weights,
            self.depress_weights,
            self.control_weights_vec,
            self.depress_weights_vec,
            self.control,
            self.depress,
            self.control_subj_ids,
            self.depress_subj_ids,
            self.labels,
        ) = self.build_dataset()

    def build_dataset(self, is_cov=False):
        """Builds the dataset."""
        no_weights = []
        missing_rois = []
        depr_files = []
        ctrl_files = []
        control_weights = np.zeros((self.n1, self.n_roi, self.n_roi))
        depress_weights = np.zeros((self.n2, self.n_roi, self.n_roi))
        control_weights_vec = np.zeros((self.n1, self.n_roi * self.n_roi))
        depress_weights_vec = np.zeros((self.n2, self.n_roi * self.n_roi))
        ctrl_subj_labels = []
        depr_subj_labels = []
        dataset_labels = []
        n_ctrl_files = 0
        n_depr_files = 0

        for i in range(len(self.labels)):
            # subject ID
            subj = "sub-" + self.labels.index.values[i]

            # build DDC files list
            files = glob.glob(
                self.DDC_path + subj + "/single_sessions/" + self.weights_file_name
            )

            for f in files:
                if os.path.exists(f):
                    # Control subjects
                    if self.labels.values[i] == '0':
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < self.n_roi:
                            if sum(sum(np.isnan(d))) < 1:
                                control_weights[n_ctrl_files, :, :] = np.asarray(
                                    pd.read_csv(f, header=None)
                                )

                                # Threshold binarization (to be replaced by bootstrap) and reshape
                                control_weights_vec[n_ctrl_files, :] = np.reshape(
                                    (
                                        abs(control_weights[n_ctrl_files, :, :])
                                        > self.thrs
                                    )
                                    * 1,
                                    (1, self.n_roi * self.n_roi),
                                )

                                ctrl_files.append(f)

                                n_ctrl_files += 1

                                ctrl_subj_labels.append(subj.split("-")[1])

                    # Depressed subjects
                    else:
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < self.n_roi:
                            if sum(sum(np.isnan(d))) < 1:
                                depress_weights[n_depr_files, :, :] = np.asarray(
                                    pd.read_csv(f, header=None)
                                )
                                # Threshold binarization (to be replaced by bootstrap) and reshape
                                depress_weights_vec[n_depr_files, :] = np.reshape(
                                    (
                                        abs(depress_weights[n_depr_files, :, :])
                                        > self.thrs
                                    )
                                    * 1,
                                    (1, self.n_roi * self.n_roi),
                                )

                                depr_files.append(f)

                                n_depr_files += 1

                                depr_subj_labels.append(subj.split("-")[1])

                else:
                    no_weights.append(f)

        control_weights = control_weights[:n_ctrl_files, :, :]
        depress_weights = depress_weights[:n_depr_files, :, :]
        control_weights_vec = control_weights_vec[:n_ctrl_files, :]
        depress_weights_vec = depress_weights_vec[:n_depr_files, :]
        control = np.reshape(
            control_weights, (len(control_weights), self.n_roi * self.n_roi)
        )
        depress = np.reshape(
            depress_weights, (len(depress_weights), self.n_roi * self.n_roi)
        )

        return (
            no_weights,
            missing_rois,
            depr_files,
            ctrl_files,
            control_weights,
            depress_weights,
            control_weights_vec,
            depress_weights_vec,
            control,
            depress,
            ctrl_subj_labels,
            depr_subj_labels,
            dataset_labels,
        )

    
    def get_binary_connections_percentage_control(self):
        sig_conn = np.reshape(
            sum(self.control_weights_vec) / np.shape(self.control_weights_vec)[0],
            (self.n_roi, self.n_roi),
        )

        return sig_conn

    def get_binary_connections_percentage_depress(self):
        sig_conn = np.reshape(
            sum(self.depress_weights_vec) / np.shape(self.depress_weights_vec)[0],
            (self.n_roi, self.n_roi),
        )

        return sig_conn

    def get_mean_ddc(self, state):
        if state == "control":
            DDC = self.control_weights
        else:
            DDC = self.depress_weights
        return np.nanmean(DDC, axis=0)

    def get_std_ddc(self, state):
        if state == "control":
            DDC = self.control_weights
        else:
            DDC = self.depress_weights

        return np.nanstd(DDC, axis=0)

    def get_network_indices(self, network_name):
        if network_name == "DMN":
            indices = self.DMN_indices
        elif network_name == "CEN":
            indices = self.CEN_indices
        elif network_name == "SN":
            indices = self.SN_indices

        return indices

    def get_network_labels(self, network_name):
        if network_name == "DMN":
            labels = self.DMN_labels
        elif network_name == "CEN":
            labels = self.CEN_labels
        elif network_name == "SN":
            labels = self.SN_labels

        return labels

    def get_network_ddc(self, network_name, state):
        indices = self.get_network_indices(network_name)

        if state == "control":
            DDC = self.control_weights
        else:
            DDC = self.depress_weights

        network_DDC = DDC[:, indices, :]
        network_DDC = network_DDC[:, :, indices]

        return network_DDC

    def get_flat_network_ddc(self, network_name, state):
        network_DDC = self.get_network_ddc(network_name, state)

        network_DDC = np.reshape(
            network_DDC, (len(network_DDC), len(network_DDC[0]) * len(network_DDC[0]))
        )

        return network_DDC

    def plot_binary_weights(self, state, plot=plt.figure(), colorbar=True):
        """plot the binary connectivity for all subjects in vector form per state"""
        if state == "control":
            weights = self.control_weights_vec.T
        else:
            weights = self.depress_weights_vec.T
        plt.imshow(weights, aspect="auto")
        if colorbar:
            plt.colorbar()
        plt.ylabel("Connections")
        plt.xlabel("Subjects")
        plt.title("{}".format(state))

    def plot_binary_weights_across_states(self):
        """plot the binary connectivity for all subjects in vector form for control and depressed."""
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        self.plot_binary_weights("control", plt, colorbar=False)

        plt.subplot(122)
        self.plot_binary_weights("depressed", plt)

    def plot_significant_connections_bar(self):
        plt.figure(figsize=(10, 5))
        plt.bar(
            np.arange(0, self.n_roi * self.n_roi, 1),
            sum(self.control_weights_vec) / np.shape(self.control_weights_vec)[0],
            label="Control",
        )
        plt.bar(
            np.arange(0, self.n_roi * self.n_roi, 1),
            sum(self.depress_weights_vec) / np.shape(self.depress_weights_vec)[0],
            label="Depress",
            alpha=0.7,
        )
        plt.xlabel("Connections#")
        plt.ylabel("counts")
        plt.legend()
        plt.grid()
        plt.ylim([0, 1])
        plt.title("Significant connections (abs(DDC)>0.1)")

        plt.savefig(
            f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_bar.svg",
            format="svg",
        )

    def plot_significant_connections_graph(self):
        """plot network graph of signifiicant connections for controls and depressed"""
        sig_connect_control = self.get_binary_connections_percentage_control()
        sig_connect_depress = self.get_binary_connections_percentage_depress()

        if self.weights_file_name.startswith("subc_"):
            thr = 0.25
        else:
            thr = 0.65

        ddc_plot = sig_connect_control > thr
        G = nx.from_numpy_array(ddc_plot)
        plt.figure(figsize=(10, 10))

        plt.subplot(121)
        nx.draw(G, np.asarray(self.positions[["x", "y"]]), with_labels=True)
        plt.title("Control")
        plt.subplot(122)
        ddc_plot = sig_connect_depress > thr
        G = nx.from_numpy_array(ddc_plot)
        plt.title("Depressed")
        nx.draw(G, np.asarray(self.positions[["x", "y"]]), with_labels=True)

        plt.savefig(
            f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_graph.svg",
            format="svg",
        )


    def plot_mean_weights(self, state, plot=plt.figure(), colorbar=True):
        """plot mean connectivity matrix per state"""
        avg = self.get_mean_ddc(state)
        limit = max(abs(np.min(avg)), abs(np.max(avg)))
        plt.imshow(avg, vmin=-limit, vmax=limit, cmap="RdBu_r")
        if colorbar:
            plt.colorbar()
        # plt.clim([-0.01, 0.01])
        plt.title("average {} weights".format(state))
        plt.xlabel("ROI #")
        plt.ylabel("ROI #"),

    def plot_mean_difference(self, plot=plt.figure()):
        """plot difference between mean connectivity matrix per state."""
        avg_ctrl = self.get_mean_ddc("control")
        avg_depr = self.get_mean_ddc("depressed")
        avg_diff = abs(avg_depr) - abs(avg_ctrl)
        limit = max(abs(np.min(avg_diff)), abs(np.max(avg_diff)))
        plt.imshow(avg_diff, vmin=-limit, vmax=limit, cmap="RdBu_r")
        plt.colorbar()
        # plt.clim([-limit, limit])
        plt.title("Difference")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

    def plot_significant_connections_matrix(
        self, colorbar=True, save_as=None, bonferroni=False, median=0, ttest=1
    ):
        # s, p = mannwhitneyu(self.control, self.depress)
        stat_diff = np.zeros(self.n_roi * self.n_roi)

        from scipy.stats import ttest_ind

        t_statistics = np.zeros((np.shape(self.control)[1], 1))
        p_values = np.zeros((np.shape(self.control)[1], 1))

        for i in range(np.shape(self.control)[1]):
            if ttest:
                t_statistics[i], p_values[i] = ttest_ind(
                    self.control[:, i], self.depress[:, i]
                )
            else:
                t_statistics[i], p_values[i] = mannwhitneyu(self.control[:, i], self.depress[:, i])

        p = p_values.reshape((self.n_roi * self.n_roi))

        if bonferroni:
            n_comp = self.control.shape[1]
            adjusted_alpha = 0.05 / n_comp
            stat_diff[np.where(p < adjusted_alpha)[0]] = 1
        else:
            stat_diff[np.where(p < 0.05)[0]] = 1

        stat_diff = np.reshape(stat_diff, (self.n_roi, self.n_roi))

        if median:
            a = np.nanmedian(self.control_weights, axis=0)
            b = np.nanmedian(self.depress_weights, axis=0)
        else:
            a = self.get_mean_ddc("control")
            b = self.get_mean_ddc("depressed")

        # diff = b - a
        diff = abs(b) - abs(a)
        diff[np.where(stat_diff == 0)] = 0

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(stat_diff, cmap="Greys")
        plt.clim([0, 1])
        plt.colorbar()
        plt.title("Statistically different connections")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.subplot(122)
        im = plt.imshow(diff, cmap="RdBu_r")
        for i in range(len(diff)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([-np.max(diff), np.max(diff)])
        plt.colorbar()
        plt.title("Statistically different connections")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        pv = np.reshape(p, (self.n_roi, self.n_roi))
        p_table_list = []
        for i in range(self.n_roi):
            for j in range(self.n_roi):
                new_row = {
                    "Areas": self.all_ROIs[i] + "-" + self.all_ROIs[j],
                    "p-value": pv[i, j],
                }
                p_table_list.append(new_row)

        p_table = pd.DataFrame(p_table_list)
        p_table.to_csv(
            "/home/acamassa/ABCD/DDC_figures/p_values_table.csv", index=False
        )

        if save_as is None:
            if bonferroni:
                plt.savefig(
                    f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix.svg",
                    format="svg",
                )
            else:
                plt.savefig(
                    f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix_Bonferroni.svg",
                    format="svg",
                )
        else:
            plt.savefig(f"{self.fig_dir}{save_as}")

        return p
    
    def plot_significant_connections_bar(self, bonferroni=False, ttest=1):

        stat_diff = np.zeros(self.n_roi * self.n_roi)

        from scipy.stats import ttest_ind

        t_statistics = np.zeros((np.shape(self.control)[1], 1))
        p_values = np.zeros((np.shape(self.control)[1], 1))
        for i in range(np.shape(self.control)[1]):
            if ttest:
                t_statistics[i], p_values[i] = ttest_ind(
                    self.control[:, i], self.depress[:, i]
                )
            else:
                t_statistics[i], p_values[i] = mannwhitneyu(self.control[:, i], self.depress[:, i])

        p = p_values.reshape((self.n_roi * self.n_roi))

        if bonferroni:
            n_comp = self.control.shape[1]
            adjusted_alpha = 0.05 / n_comp
            stat_diff[np.where(p < adjusted_alpha)[0]] = 1
        else:
            stat_diff[np.where(p < 0.05)[0]] = 1

        stat_diff = np.reshape(stat_diff, (self.n_roi, self.n_roi))
        a = self.get_mean_ddc("control")
        b = self.get_mean_ddc("depressed")
        # diff = b - a

        from operator import itemgetter

        diff = abs(b) - abs(a)
        diff[np.where(stat_diff == 0)] = 0
        diff=diff[3:,3:]
        sum_diff = diff.sum(axis=0)

        indices = np.argsort(sum_diff)[::-1]
        new_labels=[]
        labels=self.all_ROIs[3:]    

        for i in indices:
            new_labels.append(labels[i])

        fig=plt.figure(figsize=(20,5))
        colors = plt.cm.Reds_r(np.linspace(0, 1, len(diff)))
        plt.bar(np.arange(len(diff[1:])),sum_diff[indices][1:],color=colors)
        plt.xticks(np.arange(len(new_labels[1:])), new_labels[1:], rotation="vertical")

        plt.savefig(
            f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_barplot.svg",
            format="svg",
        )

        # # plot on the brain
        coord_list = np.asarray(self.positions[["x", "y", "z"]])[3:,:]

        fig=plt.figure(figsize=(15,7))
        plotting.plot_markers(
            sum_diff,
            coord_list,
            node_cmap="Reds",
            figure=fig,
            # node_vmin=-2e-18,
            # node_vmax=2e-18,
            node_size=sum_diff/10,
            title="Depressed-Controls",
        )
    
        plt.savefig("/home/acamassa/ABCD/DDC_figures/Depr-Controls_FC_map.svg")
        plt.show()

        

         

    def plot_means_connectivity_matrices(self, colorbar=False):
        """plot mean connectivity matrix for controls and depressed"""
        plt.figure(figsize=(10, 5))

        plt.subplot(131)
        self.plot_mean_weights("control", plt, colorbar)

        plt.subplot(132)
        self.plot_mean_weights("depressed", plt, colorbar)

        plt.subplot(133)
        self.plot_mean_difference(plt)

        plt.tight_layout()

        # plt.colorbar()

    def plot_means_std_matrices(self, save_as=None,cmap="Reds", median=0):
        if median:
            avg_ddc_ctrl=np.nanmedian(self.control_weights, axis=0)
            avg_ddc_depr=np.nanmedian(self.depress_weights, axis=0)
        else:
            avg_ddc_ctrl = self.get_mean_ddc("control")
            avg_ddc_depr = self.get_mean_ddc("depressed")

        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        im = plt.imshow(avg_ddc_ctrl, cmap="RdBu_r")
        for i in range(len(avg_ddc_ctrl)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        # Get control values
        cbar_min = min(avg_ddc_ctrl.flatten())
        cbar_max = max(avg_ddc_ctrl.flatten())

        # Get depressed values
        cbar_min = min(cbar_min, min(avg_ddc_depr.flatten()))
        cbar_max = max(cbar_max, max(avg_ddc_depr.flatten()))
        plt.clim([cbar_min, cbar_max])
        # plt.clim([-400, 400])
        plt.colorbar()
        plt.title("avg DDC control")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        plt.subplot(222)
        im = plt.imshow(avg_ddc_depr, cmap="RdBu_r")
        for i in range(len(avg_ddc_depr)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([cbar_min, cbar_max])
        plt.title("avg DDC depr")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.colorbar()

        plt.subplot(223)
        std_ddc_ctrl = self.get_std_ddc("control")
        im = plt.imshow(std_ddc_ctrl, cmap=cmap)
        plt.colorbar()
        # plt.clim([0, 10000])
        plt.title("std DDC control")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        
        std_ddc_depr = self.get_std_ddc("depressed")

        # Get control values
        cbar_min = min(std_ddc_ctrl.flatten())
        cbar_max = max(std_ddc_ctrl.flatten())
        # Get depressed values
        cbar_min = min(cbar_min, min(std_ddc_depr.flatten()))
        cbar_max = max(cbar_max, max(std_ddc_depr.flatten()))
        plt.clim([cbar_min, cbar_max])

        plt.subplot(224)
        im = plt.imshow(std_ddc_depr, cmap=cmap)
        plt.colorbar()
        plt.clim([cbar_min, cbar_max])
        # plt.clim([0, 10000])
        plt.title("std DDC depr")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        if save_as is None:
            plt.savefig(
                f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_mean_std.svg",
                format="svg",
            )
        else:
            plt.savefig(f"{self.fig_dir}{save_as}")

    def plot_random_matrices(self, state):
        """plot 25 random DDC matrices"""
        if state == "control":
            DDC = self.control_weights
        else:
            DDC = self.depress_weights
        n = random.sample(range(len(DDC)), 25)
        fig = plt.figure(figsize=(10, 10))
        k = 0
        for i in n:
            ax = fig.add_subplot(5, 5, k + 1)
            plt.imshow(DDC[i, :, :], cmap="RdBu_r")
            # plt.clim([-0.1, 0.1])
            plt.colorbar()
            plt.title(str(i))

            plt.axis("off")
            k = k + 1

    def subset_fc(self, fc, include):
        """get subnetwork connectivity"""
        a = fc[include, :]
        a = a[:, include]
        return a

    def plot_network_heatmap(self, network_name, save_as=None, bonferroni=False, scaling=0, median=0, ttest=1):
        """plot binary matrix of significant connections for a specific subnetwork"""
        plt.figure(figsize=(15, 5))

        if scaling:
            scaled_c, scaled_d = self.standard_scaling()
        else:
            scaled_c=self.control_weights
            scaled_d=self.depress_weights

        indices = self.get_network_indices(network_name)
        labels = self.get_network_labels(network_name)

        network_ctrl = scaled_c[:, indices, :]
        network_ctrl = network_ctrl[:, :, indices]
        network_depr = scaled_d[:, indices, :]
        network_depr = network_depr[:, :, indices]

        # network_ctrl = self.get_network_ddc(network_name, "control")
        # network_depr = self.get_network_ddc(network_name, "depressed")

        # Get control values
        # avg_ctrl = self.get_mean_ddc("control")

        if median:
            avg_ctrl = np.nanmedian(self.control_weights, axis=0)
        else:
            avg_ctrl = np.nanmean(self.control_weights, axis=0)

        ctrl_fc = self.subset_fc(avg_ctrl, indices)
        cbar_min = min(ctrl_fc.flatten())
        cbar_max = max(ctrl_fc.flatten())

        # Get depressed values
        # avg_depr = self.get_mean_ddc("depressed")
        if median:
            avg_depr = np.nanmedian(self.depress_weights, axis=0)
        else:
            avg_depr = np.nanmean(self.depress_weights, axis=0)

        depr_fc = self.subset_fc(avg_depr, indices)
        cbar_min = min(cbar_min, min(depr_fc.flatten()))
        cbar_max = max(cbar_max, max(depr_fc.flatten()))

        # Plot control
        plt.subplot(131)
        plt.title("{} control".format(network_name))
        im = plt.imshow(ctrl_fc, cmap="RdBu_r")
        plt.colorbar()
        # Add gray boxes for self-connections
        for i in range(len(ctrl_fc)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([cbar_min, cbar_max])
        plt.yticks(np.arange(len(indices)), labels)
        plt.xticks(np.arange(len(indices)), labels, rotation="vertical")

        # Plot depressed
        plt.subplot(132)
        plt.title("{} depressed".format(network_name))
        im = plt.imshow(depr_fc, cmap="RdBu_r")
        plt.colorbar()
        # Add gray boxes for self-connections
        for i in range(len(depr_fc)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([cbar_min, cbar_max])
        plt.yticks(np.arange(len(indices)), labels)
        plt.xticks(np.arange(len(indices)), labels, rotation="vertical")

        # non parametric statistical test for independent variables
        # _, p = mannwhitneyu(network_ctrl, network_depr)
        from scipy.stats import ttest_ind

        c = network_ctrl.reshape(len(network_ctrl), -1)
        d = network_depr.reshape(len(network_depr), -1)

        t_statistics = np.zeros((np.shape(c)[1], 1))
        p_values = np.zeros((np.shape(c)[1], 1))
        for i in range(np.shape(c)[1]):
            if ttest:
                t_statistics[i], p_values[i] = ttest_ind(c[:, i], d[:, i])
            else:
                t_statistics[i], p_values[i] = mannwhitneyu(c[:, i], d[:, i])

        p = p_values.reshape(np.shape(network_ctrl)[1:])
        # absolute difference
        diff= abs(depr_fc) - abs(ctrl_fc)

        if bonferroni:
            n_comp = network_ctrl.shape[1]
            adjusted_alpha = 0.05 / n_comp
            diff[np.where(p > adjusted_alpha)] = 0
        else:
            diff[np.where(p > 0.05)] = 0

        # Plot if there are significant differences
        # if sum(sum(diff)) > 0:
        plt.subplot(133)
        im = plt.imshow(diff, cmap="RdBu_r")
        # if network_name == "CEN":
        #     plt.clim([-0.0002, 0.0002])
        # else:
        plt.clim([-np.max(diff), np.max(diff)])

        plt.colorbar()

        for i in range(len(diff)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
                        
        for i in range(len(ctrl_fc)):
            for j in range(len(ctrl_fc)):
                if diff[i,j]!=0:
                    if i!=j:
                        if np.sign(ctrl_fc[i,j]) != np.sign(depr_fc[i,j]):
                            plt.scatter(j, i, marker='*', color='k', s=50) 
                            # plt.xlim([0,len(ctrl_fc)])
                            # plt.ylim([0,len(ctrl_fc)])




        # plt.colorbar()
        plt.yticks(np.arange(len(indices)), labels)
        plt.xticks(np.arange(len(indices)), labels, rotation="vertical")
        plt.title("Statisticaly different fc")



        if save_as is not None:
            plt.savefig(f"{self.fig_dir}{save_as}")
        else:
            if bonferroni:
                plt.savefig(
                    f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix"
                    + str(network_name)
                    + "Bonferroni.svg",
                    format="svg",
                )
            else:
                plt.savefig(
                    f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix"
                    + str(network_name)
                    + ".svg",
                    format="svg",
                )
        # return diff

    def plot_network_significant_connections_graph(self, network_name):
        """plot network graph only significantly different connections for a specific subnetwork"""
        indices = self.get_network_indices(network_name)
        labels = self.get_network_labels(network_name)

        # non parametric statistical test for independent variables
        network_control = self.get_network_ddc(network_name, "control")
        network_depr = self.get_network_ddc(network_name, "depressed")
        _, p = mannwhitneyu(network_control, network_depr)

        G = nx.from_numpy_array(p < 0.05)

        coord_list = np.asarray(self.positions[["x", "y"]])
        plt.figure(figsize=(5, 5))
        labeldict = {}
        for i in range(len(indices)):
            labeldict[i] = labels[i]
        nx.draw(
            G,
            coord_list[indices, :],
            node_color="orange",
            with_labels=True,
            labels=labeldict,
        )
        plt.title("{} significantly different connections".format(network_name))

        plt.savefig(
            f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_network"
            + str(network_name)
            + ".svg",
            format="svg",
        )

    def plot_network_connectivity_graph(self, network_name, state):
        """plot network graph on brain template of connections for a specific subnetwork and state"""
        indices = self.get_network_indices(network_name)

        coord_list = np.asarray(self.positions[["x", "y", "z"]])
        avg = self.get_mean_ddc(state)
        a = self.subset_fc(avg, indices)
        if self.weights_file_name.startswith("filt_Cov"):
            ev = 0.5
        elif self.weights_file_name.startswith("Reg"):
            ev = 0.005
        elif self.weights_file_name.startswith("subc_Reg"):
            ev = 0.0025
        display = plotting.plot_connectome(
            a,
            coord_list[indices, :],
            node_color="r",
            edge_cmap=None,
            #             edge_vmax=ev,
            title="{} {}".format(state, network_name),
            colorbar=True,
        )

        plotting.show()

        display.savefig(
            f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_network_connectivity"
            + str(network_name)
            + str(state)
            + ".svg"
        )

    def plot_network_connectivity_graph_diff(
        self, network_name, ev, save_as=None, bonferroni=False, scaling=0, ttest=1
    ):
        """plot differences between ctrl and depressed network graph on brain template for a specific subnetwork"""
    
        if scaling:
            scaled_c, scaled_d = self.standard_scaling()
        else:
            scaled_c=self.control_weights
            scaled_d=self.depress_weights

        indices = self.get_network_indices(network_name)
        labels = self.get_network_labels(network_name)

        network_ctrl = scaled_c[:, indices, :]
        network_ctrl = network_ctrl[:, :, indices]
        network_depr = scaled_d[:, indices, :]
        network_depr = network_depr[:, :, indices]

        coord_list = np.asarray(self.positions[["x", "y", "z"]])

        # Get control values
        avg_ctrl = self.get_mean_ddc("control")
        ctrl_fc = self.subset_fc(avg_ctrl, indices)
        cbar_min = min(ctrl_fc.flatten())
        cbar_max = max(ctrl_fc.flatten())

        # Get depressed values
        avg_depr = self.get_mean_ddc("depressed")
        depr_fc = self.subset_fc(avg_depr, indices)
        cbar_min = min(cbar_min, min(depr_fc.flatten()))
        cbar_max = max(cbar_max, max(depr_fc.flatten()))


        from scipy.stats import ttest_ind
        if ttest:
            _, p = ttest_ind(network_ctrl, network_depr)
        else:
            _, p = mannwhitneyu(network_ctrl, network_depr)

        diff = abs(depr_fc) - abs(ctrl_fc)

        if bonferroni:
            n_comp = network_ctrl.shape[1]
            adjusted_alpha = 0.05 / n_comp
            diff[np.where(p > adjusted_alpha)] = 0
        else:
            diff[np.where(p > 0.05)] = 0

        display = plotting.plot_connectome(
            diff,
            coord_list[indices, :],
            node_color="k",
            edge_cmap="RdBu_r",
            edge_vmax=cbar_max,
            edge_kwargs=dict(lw=4),
            title="{}".format(network_name),
            colorbar=True,
        )

        plotting.show()

        if save_as is None:
            display.savefig(
                f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_diff_network_connectivity"
                + str(network_name)
                + ".svg"
            )
        else:
            plt.savefig(f"{self.fig_dir}{save_as}")

    #         else:
    #             print("no different connections to plot")

    def plot_connectivity_graph(self, state):
        """plot network graph on brain template of all connections for a specific state"""

        coord_list = np.asarray(self.positions[["x", "y", "z"]])

        avg = self.get_mean_ddc(state)
        display = plotting.plot_connectome(
            avg,
            coord_list,
            edge_cmap=None,
            edge_threshold="98%",
            title="{}".format(state),
        )
        plotting.show()

        display.savefig(
            f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_network_connectivity"
            + str(state)
            + ".svg"
        )

    def plot_interactive_connectivity_graph(self, state):
        """interactive 3D plot network graph on brain template of all connections for a specific state"""

        coord_list = np.asarray(self.positions[["x", "y", "z"]])
        avg = self.get_mean_ddc(state)

        view = plotting.view_connectome(
            avg,
            coord_list,
            edge_cmap=None,
            edge_threshold="95%",
            title="{}".format(state),
            symmetric_cmap=False,
        )

        return view

    def plot_interactive_connectivity_graph_diff(self):
        s, p = mannwhitneyu(self.control, self.depress)
        stat_diff = np.zeros(self.n_roi * self.n_roi)
        stat_diff[np.where(p < 0.05)[0]] = 1
        stat_diff = np.reshape(stat_diff, (self.n_roi, self.n_roi))
        a = self.get_mean_ddc("control")
        b = self.get_mean_ddc("depressed")
        diff= abs(b) - abs(a)
        diff[np.where(stat_diff == 0)] = 0

        coord_list = np.asarray(self.positions[["x", "y", "z"]])

        view = plotting.view_connectome(
            diff,
            coord_list,
            edge_cmap="RdBu_r",
            edge_threshold=50,
            symmetric_cmap=False,
            colorbar=True,
            node_color="k",
        )
        # pd.DataFrame(diff).to_csv('/home/acamassa/ABCD/DDC/figures/stat_diff_matrix.csv')
        return view

    def standard_scaling(self):
        "standard scaling the whole dataset of FC values between -1 and 1"

        c = self.control_weights.reshape(len(self.control_weights), -1)
        d = self.depress_weights.reshape(len(self.depress_weights), -1)

        o_max = np.max((int(np.max(c)), int(np.max(d))))
        o_min = np.min((int(np.min(c)), int(np.min(d))))
        # Define the target range for the scaled data
        target_min = -1
        target_max = 1

        # Perform Min-Max scaling with custom range
        scaled_c = (c - o_min) * (target_max - target_min) / (
            o_max - o_min
        ) + target_min
        scaled_d = (d - o_min) * (target_max - target_min) / (
            o_max - o_min
        ) + target_min

        scaled_c = scaled_c.reshape(len(self.control_weights), self.n_roi, self.n_roi)
        scaled_d = scaled_d.reshape(len(self.depress_weights), self.n_roi, self.n_roi)

        return scaled_c, scaled_d




    def plot_connection_probDistr(self, x=None, y=None, save_as=None, scaling=0, ttest=1):
        "plotting the distribution for the selected connection"
        "x and y can be int = index of the desired ROIs or str + name of the desired ROIs to compare"
        import seaborn as sns
        from scipy.stats import ttest_ind

        if scaling:
            scaled_c, scaled_d = self.standard_scaling()
        else:
            scaled_c = self.control_weights
            scaled_d = self.depress_weights

        if isinstance(x, str):
            x = self.all_ROIs.index(x)
            y = self.all_ROIs.index(y)

        # Create histograms
        plt.figure(figsize=(12, 6))

        sns.histplot( np.log(np.abs(scaled_c[:, x, y])) , kde=True, color='blue', label='control', log_scale=(False, False))
        sns.histplot(np.log(np.abs(scaled_d[:, x, y])), kde=True, color='orange', label='depressed', log_scale=(False, False))

        # Add mean and median vertical bars
        mean_c = np.nanmean(np.log(np.abs(scaled_c[:, x, y])))
        mean_d = np.nanmean(np.log(np.abs(scaled_d[:, x, y])))
        median_c = np.nanmedian(np.log(np.abs(scaled_c[:, x, y])))
        median_d = np.nanmedian(np.log(np.abs(scaled_d[:, x, y])))

        plt.axvline(x=mean_c, color='cyan', linestyle='dashed', linewidth=2, label='Mean (Control)')
        plt.axvline(x=mean_d, color='red', linestyle='dashed', linewidth=2, label='Mean (Depressed)')
        plt.axvline(x=median_c, color='cyan', linestyle='dotted', linewidth=2, label='Median (Control)')
        plt.axvline(x=median_d, color='red', linestyle='dotted', linewidth=2, label='Median (Depressed)')

        plt.title(f"{self.all_ROIs[x]}:{self.all_ROIs[y]}")
        plt.legend()

        # Perform t-test
        if ttest:
            t_stat, p_value = ttest_ind(scaled_c[:, x, y], scaled_d[:, x, y], nan_policy='omit')
        else:
            t_stat, p_value = mannwhitneyu(scaled_c[:, x, y], scaled_d[:, x, y])

        plt.annotate(f'p-value: {p_value:.4f}', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.show()


    # def plot_connection_probDistr(self, x=None, y=None, save_as=None, scaling=0, ttest=1):
    #     "plotting the distribution for the selected connection"
    #     "x and y can be int = index of the desired ROIs or str + name of the desired ROIs to compare"
    #     import seaborn as sns
    #     from scipy.stats import ttest_ind

    #     if scaling:
    #         scaled_c, scaled_d = self.standard_scaling()
    #     else:
    #         scaled_c=self.control_weights
    #         scaled_d=self.depress_weights

    #     if isinstance(x, str):
    #         x = self.all_ROIs.index(x)
    #         y = self.all_ROIs.index(y)
            
    #     # Create histograms
    #     plt.figure(figsize=(12, 6))

    #     sns.histplot(scaled_c[:, x, y], kde=True, color='blue', label='control',log_scale=(True, False))
    #     sns.histplot(scaled_d[:, x, y], kde=True, color='orange', label='depressed',log_scale=(True, False))
    #     plt.title(f"{self.all_ROIs[x]}:{self.all_ROIs[y]}")
    #     plt.legend() 
    #     # Perform t-test
    #     if ttest:
    #         t_stat, p_value = ttest_ind(scaled_c[:, x, y], scaled_d[:, x, y])
    #     else:
    #         t_stat, p_value = mannwhitneyu(scaled_c[:, x, y], scaled_d[:, x, y])

    #     plt.annotate(f'p-value: {p_value:.4f}', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

################# SEX ANALYSIS ############################################# SEX ANALYSIS ############################
    
    def get_controls_sex(self, labels):

        labels=labels[1:]
        labels_c=labels[labels[4]=='0']
        
        control_M=[]
        control_F=[]

        for i in range(len(labels_c)):
            # subject ID
            subj = "sub-" + labels_c.index.values[i]
            # build DDC files list
            files = glob.glob(
                self.DDC_path + subj + "/single_sessions/" + self.weights_file_name
            )
            # print(subj)
            for f in files:
                if os.path.exists(f):
                    # print(f)
                    # sex
                    if labels_c.values[i,0] == 'M':
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < 98:
                            if sum(sum(np.isnan(d))) < 1:
                                control_M.append(d)
                    elif labels_c.values[i,0] == 'F':
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < 98:
                            if sum(sum(np.isnan(d))) < 1:
                                control_F.append(d)

        control_M=np.asarray(control_M)
        control_F=np.asarray(control_F)
        return control_M, control_F
    
    def get_depr_sex(self, labels):

        labels=labels[1:]
        labels_c=labels[labels[4]=='1']
        
        control_M=[]
        control_F=[]

        for i in range(len(labels_c)):
            # subject ID
            subj = "sub-" + labels_c.index.values[i]
            # build DDC files list
            files = glob.glob(
                self.DDC_path + subj + "/single_sessions/" + self.weights_file_name
            )
            # print(subj)
            for f in files:
                if os.path.exists(f):
                    # print(f)
                    # sex
                    if labels_c.values[i,0] == 'M':
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < 98:
                            if sum(sum(np.isnan(d))) < 1:
                                control_M.append(d)
                    elif labels_c.values[i,0] == 'F':
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < 98:
                            if sum(sum(np.isnan(d))) < 1:
                                control_F.append(d)

        control_M=np.asarray(control_M)
        control_F=np.asarray(control_F)
        return control_M, control_F

    def plot_significant_sex_diff(
        self, control_M, control_F, condition, colorbar=True, save_as=None, bonferroni=False, median=0, ttest=1
    ):
        if median==0:
            a = np.nanmean(control_M,axis=0)
            b = np.nanmean(control_F,axis=0)
        else:
            a = np.nanmedian(control_M,axis=0)
            b = np.nanmedian(control_F,axis=0)

        control_M = np.reshape(
            control_M, (len(control_M), self.n_roi * self.n_roi)
        )
        control_F = np.reshape(
            control_F, (len(control_F), self.n_roi * self.n_roi)
        )

        stat_diff = np.zeros(self.n_roi * self.n_roi)

        from scipy.stats import ttest_ind

        t_statistics = np.zeros((np.shape(control_M)[1], 1))
        p_values = np.zeros((np.shape(control_M)[1], 1))

        for i in range(np.shape(control_M)[1]):
            if ttest:
                t_statistics[i], p_values[i] = ttest_ind(
                    control_M[:, i], control_F[:, i]
                )
            else:
                t_statistics[i], p_values[i] = mannwhitneyu(
                    control_M[:, i], control_F[:, i]
                )


        p = p_values.reshape((self.n_roi * self.n_roi))

        if bonferroni:
            n_comp = self.control.shape[1]
            adjusted_alpha = 0.05 / n_comp
            stat_diff[np.where(p < adjusted_alpha)[0]] = 1
        else:
            stat_diff[np.where(p < 0.05)[0]] = 1

        stat_diff = np.reshape(stat_diff, (self.n_roi, self.n_roi))
        # diff = b - a
        diff = abs(b) - abs(a)
        diff[np.where(stat_diff == 0)] = 0

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(stat_diff, cmap="Greys")
        plt.clim([0, 1])
        plt.colorbar()
        if condition==[0,0]:
            plt.title("Sex different connections in controls")
        if condition==[1,1]:
            plt.title("Sex different connections in depressed")
        if condition==[0,1]:
            plt.title("Different connections in Males")
        if condition==[1,0]:
            plt.title("Different connections in Females")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.subplot(122)
        im = plt.imshow(diff, cmap="RdBu_r")
        for i in range(len(diff)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([-np.max(diff), np.max(diff)])
        plt.colorbar()
        if condition==[0,0]:
            plt.title("Sex different connections in controls")
        if condition==[1,1]:
            plt.title("Sex different connections in depressed")
        if condition==[0,1]:
            plt.title("Different connections in Males")
        if condition==[1,0]:
            plt.title("Different connections in Females")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        pv = np.reshape(p, (self.n_roi, self.n_roi))
        p_table_list = []
        for i in range(self.n_roi):
            for j in range(self.n_roi):
                new_row = {
                    "Areas": self.all_ROIs[i] + "-" + self.all_ROIs[j],
                    "p-value": pv[i, j],
                }
                p_table_list.append(new_row)

        p_table = pd.DataFrame(p_table_list)
        p_table.to_csv(
            "/home/acamassa/ABCD/DDC_figures/p_values_table_sex"+ str(condition)+".csv", index=False
        )

        plt.savefig(f"{self.fig_dir}{save_as}")

        return stat_diff
    
    def plot_significant_sex_diff_distribution(self, stat_diff, control_M, control_F, ttest=1):

        a=np.where(stat_diff==1)
        for i in range(np.shape(a)[1]):

            x = a[0][i]
            y = a[1][i]
            
            # Create histograms
            plt.figure(figsize=(12, 6))

            sns.histplot(control_M[:, x, y], kde=True, color='cyan', label='Males',log_scale=(True, False))
            sns.histplot(control_F[:, x, y], kde=True, color='magenta', label='Females',log_scale=(True, False))
            plt.title(f"{self.all_ROIs[x]}:{self.all_ROIs[y]}")
            plt.legend() 
            # Perform t-test
            if ttest:
                _ , p_value = ttest_ind(control_M[:, x, y], control_F[:, x, y])
            else:
                _ , p_value = mannwhitneyu(control_M[:, x, y], control_F[:, x, y])

            plt.annotate(f'p-value: {p_value:.4f}', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


        return a
    
   
    def plot_significant_sex_diff_distribution(self, stat_diff, control_M, control_F, ttest=1):

        a = np.where(stat_diff == 1)
        
        # Filter ROIs based on the condition
        valid_rois = [roi for roi in self.all_ROIs if roi not in ['CSF', '3V', '4V']]
        
        # Filter significant differences based on valid ROIs
        valid_diff_indices = [(x, y) for x, y in zip(a[0], a[1]) if self.all_ROIs[x] in valid_rois and self.all_ROIs[y] in valid_rois]
        
        # Calculate the number of subplots based on the number of valid significant differences
        num_plots = len(valid_diff_indices)
        num_cols = 4  # You can adjust the number of columns in the grid
        num_rows = (num_plots + num_cols - 1) // num_cols

        # Create a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))

        for i, (x, y) in enumerate(valid_diff_indices):
            ax = axes.flatten()[i]

            # Create histograms
            sns.histplot(control_M[:, x, y], kde=True, color='cyan', label='Males', log_scale=(True, False), ax=ax)
            sns.histplot(control_F[:, x, y], kde=True, color='magenta', label='Females', log_scale=(True, False), ax=ax)
            ax.set_title(f"{self.all_ROIs[x]}:{self.all_ROIs[y]}")
            ax.legend() 

            # Stats
            if ttest:
                _ , p_value = ttest_ind(control_M[:, x, y], control_F[:, x, y])
            else:
                _ , p_value = mannwhitneyu(control_M[:, x, y], control_F[:, x, y])

            ax.annotate(f'p-value: {p_value:.4f}', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        return a



    def plot_network_heatmap_sex(self, control_M, depr_M, network_name, sex, save_as=None, bonferroni=False, scaling=0, median=0,ttest=1):
        """plot binary matrix of significant connections for a specific subnetwork"""
        plt.figure(figsize=(15, 5))

        scaled_c=control_M
        scaled_d=depr_M

        indices = self.get_network_indices(network_name)
        labels = self.get_network_labels(network_name)

        network_ctrl = scaled_c[:, indices, :]
        network_ctrl = network_ctrl[:, :, indices]
        network_depr = scaled_d[:, indices, :]
        network_depr = network_depr[:, :, indices]

        # network_ctrl = self.get_network_ddc(network_name, "control")
        # network_depr = self.get_network_ddc(network_name, "depressed")

        # Get control values
        # avg_ctrl = self.get_mean_ddc("control")
        if median==0:
            avg_ctrl = np.nanmean(self.control_weights, axis=0)
        else:
            avg_ctrl = np.nanmedian(self.control_weights, axis=0)

        ctrl_fc = self.subset_fc(avg_ctrl, indices)
        cbar_min = min(ctrl_fc.flatten())
        cbar_max = max(ctrl_fc.flatten())

        # Get depressed values
        # avg_depr = self.get_mean_ddc("depressed")
        if median==0:
            avg_depr = np.nanmean(self.depress_weights, axis=0)
        else:
            avg_depr = np.nanmedian(self.depress_weights, axis=0)

        depr_fc = self.subset_fc(avg_depr, indices)
        cbar_min = min(cbar_min, min(depr_fc.flatten()))
        cbar_max = max(cbar_max, max(depr_fc.flatten()))

        # Plot control
        plt.subplot(131)
        plt.title("{} {} control".format(network_name, sex))
        im = plt.imshow(ctrl_fc, cmap="RdBu_r")
        plt.colorbar()
        # Add gray boxes for self-connections
        for i in range(len(ctrl_fc)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([cbar_min, cbar_max])
        plt.yticks(np.arange(len(indices)), labels)
        plt.xticks(np.arange(len(indices)), labels, rotation="vertical")

        # Plot depressed
        plt.subplot(132)
        plt.title("{} {} depressed".format(network_name, sex))
        im = plt.imshow(depr_fc, cmap="RdBu_r")
        plt.colorbar()
        # Add gray boxes for self-connections
        for i in range(len(depr_fc)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([cbar_min, cbar_max])
        plt.yticks(np.arange(len(indices)), labels)
        plt.xticks(np.arange(len(indices)), labels, rotation="vertical")

        # non parametric statistical test for independent variables
        # _, p = mannwhitneyu(network_ctrl, network_depr)
        from scipy.stats import ttest_ind

        c = network_ctrl.reshape(len(network_ctrl), -1)
        d = network_depr.reshape(len(network_depr), -1)

        t_statistics = np.zeros((np.shape(c)[1], 1))
        p_values = np.zeros((np.shape(c)[1], 1))
        for i in range(np.shape(c)[1]):
            if ttest:
                t_statistics[i], p_values[i] = ttest_ind(c[:, i], d[:, i])
            else:
                t_statistics[i], p_values[i] = mannwhitneyu(c[:, i], d[:, i])

        p = p_values.reshape(np.shape(network_ctrl)[1:])

        diff= abs(depr_fc) - abs(ctrl_fc)

        if bonferroni:
            n_comp = network_ctrl.shape[1]
            adjusted_alpha = 0.05 / n_comp
            diff[np.where(p > adjusted_alpha)] = 0
        else:
            diff[np.where(p > 0.05)] = 0

        # Plot if there are significant differences
        # if sum(sum(diff)) > 0:
        plt.subplot(133)
        im = plt.imshow(diff, cmap="RdBu_r")
        # if network_name == "CEN":
        #     plt.clim([-0.0002, 0.0002])
        # else:
        plt.clim([-np.max(diff), np.max(diff)])

        plt.colorbar()

        for i in range(len(diff)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
                        
        for i in range(len(ctrl_fc)):
            for j in range(len(ctrl_fc)):
                if diff[i,j]!=0:
                    if i!=j:
                        if np.sign(ctrl_fc[i,j]) != np.sign(depr_fc[i,j]):
                            plt.scatter(j, i, marker='*', color='k', s=50) 
                            # plt.xlim([0,len(ctrl_fc)])
                            # plt.ylim([0,len(ctrl_fc)])




        # plt.colorbar()
        plt.yticks(np.arange(len(indices)), labels)
        plt.xticks(np.arange(len(indices)), labels, rotation="vertical")
        plt.title("Statisticaly different fc")



        if save_as is not None:
            plt.savefig(f"{self.fig_dir}{save_as}")
        else:
            if bonferroni:
                plt.savefig(
                    f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix"
                    + str(network_name)
                    + "Bonferroni.svg",
                    format="svg",
                )
            else:
                plt.savefig(
                    f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix"
                    + str(network_name)
                    + ".svg",
                    format="svg",
                )
        # return diff

################# HANDEDNESS ANALYSIS ############################################# HANDEDNESS ANALYSIS ############################
   
    def get_controls_hand(self, labels):

        labels=labels[1:]
        labels_c=labels[labels[4]=='0']
        
        control_R=[]
        control_L=[]

        for i in range(len(labels_c)):
            # subject ID
            subj = "sub-" + labels_c.index.values[i]
            # build DDC files list
            files = glob.glob(
                self.DDC_path + subj + "/single_sessions/" + self.weights_file_name
            )
            # print(subj)
            for f in files:
                if os.path.exists(f):
                    # print(f)
                    # sex
                    if labels_c.values[i,2] == 'R':
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < 98:
                            if sum(sum(np.isnan(d))) < 1:
                                control_R.append(d)
                    elif labels_c.values[i,2] == 'L':
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < 98:
                            if sum(sum(np.isnan(d))) < 1:
                                control_L.append(d)

        control_R=np.asarray(control_R)
        control_L=np.asarray(control_L)
        return control_R, control_L
    
    def get_depr_hand(self, labels):

        labels=labels[1:]
        labels_c=labels[labels[4]=='1']
        
        control_R=[]
        control_L=[]

        for i in range(len(labels_c)):
            # subject ID
            subj = "sub-" + labels_c.index.values[i]
            # build DDC files list
            files = glob.glob(
                self.DDC_path + subj + "/single_sessions/" + self.weights_file_name
            )
            # print(subj)
            for f in files:
                if os.path.exists(f):
                    # print(f)
                    # sex
                    if labels_c.values[i,2] == 'R':
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < 98:
                            if sum(sum(np.isnan(d))) < 1:
                                control_R.append(d)
                    elif labels_c.values[i,2] == 'L':
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < 98:
                            if sum(sum(np.isnan(d))) < 1:
                                control_L.append(d)

        depr_R=np.asarray(control_R)
        depr_L=np.asarray(control_L)
        return depr_R, depr_L
    
    def plot_significant_hand_diff(
        self, control_R, control_L, condition, colorbar=True, save_as=None, bonferroni=False, median=0, ttest=1
    ):
        if median==0:
            a = np.nanmean(control_R,axis=0)
            b = np.nanmean(control_L,axis=0)
        else:
            a = np.nanmedian(control_R,axis=0)
            b = np.nanmedian(control_L,axis=0)

        control_R = np.reshape(
            control_R, (len(control_R), self.n_roi * self.n_roi)
        )
        control_L = np.reshape(
            control_L, (len(control_L), self.n_roi * self.n_roi)
        )
        # s, p = mannwhitneyu(self.control, self.depress)
        stat_diff = np.zeros(self.n_roi * self.n_roi)

        from scipy.stats import ttest_ind

        t_statistics = np.zeros((np.shape(control_R)[1], 1))
        p_values = np.zeros((np.shape(control_R)[1], 1))

        for i in range(np.shape(control_R)[1]):
            if ttest:
                t_statistics[i], p_values[i] = ttest_ind(
                    control_R[:, i], control_L[:, i]
                )
            else:
                t_statistics[i], p_values[i] = mannwhitneyu(control_R[:, i], control_L[:, i])

        p = p_values.reshape((self.n_roi * self.n_roi))

        if bonferroni:
            n_comp = self.control.shape[1]
            adjusted_alpha = 0.05 / n_comp
            stat_diff[np.where(p < adjusted_alpha)[0]] = 1
        else:
            stat_diff[np.where(p < 0.05)[0]] = 1

        stat_diff = np.reshape(stat_diff, (self.n_roi, self.n_roi))
        # diff = b - a
        diff = abs(b) - abs(a)
        diff[np.where(stat_diff == 0)] = 0

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(stat_diff, cmap="Greys")
        plt.clim([0, 1])
        plt.colorbar()
        if condition==[0,0]:
            plt.title("Hand different connections in controls")
        if condition==[1,1]:
            plt.title("Hand different connections in depressed")
        if condition==[0,1]:
            plt.title("Different connections in R-handed")
        if condition==[1,0]:
            plt.title("Different connections in L-handed")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.subplot(122)
        im = plt.imshow(diff, cmap="RdBu_r")
        for i in range(len(diff)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([-np.max(diff), np.max(diff)])
        plt.colorbar()
        if condition==[0,0]:
            plt.title("Hand different connections in controls")
        if condition==[1,1]:
            plt.title("Hand different connections in depressed")
        if condition==[0,1]:
            plt.title("Different connections in R-handed")
        if condition==[1,0]:
            plt.title("Different connections in L-handed")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        pv = np.reshape(p, (self.n_roi, self.n_roi))
        p_table_list = []
        for i in range(self.n_roi):
            for j in range(self.n_roi):
                new_row = {
                    "Areas": self.all_ROIs[i] + "-" + self.all_ROIs[j],
                    "p-value": pv[i, j],
                }
                p_table_list.append(new_row)

        p_table = pd.DataFrame(p_table_list)
        p_table.to_csv(
            "/home/acamassa/ABCD/DDC_figures/p_values_table_hand"+ str(condition)+".csv", index=False
        )

        plt.savefig(f"{self.fig_dir}{save_as}")

        return stat_diff
    

    def plot_network_heatmap_hand(self, control_M, depr_M, network_name, hand, save_as=None, bonferroni=False, median=0, ttest=1):
            """plot binary matrix of significant connections for a specific subnetwork"""
            plt.figure(figsize=(15, 5))

            scaled_c=control_M
            scaled_d=depr_M

            indices = self.get_network_indices(network_name)
            labels = self.get_network_labels(network_name)

            network_ctrl = scaled_c[:, indices, :]
            network_ctrl = network_ctrl[:, :, indices]
            network_depr = scaled_d[:, indices, :]
            network_depr = network_depr[:, :, indices]

            # network_ctrl = self.get_network_ddc(network_name, "control")
            # network_depr = self.get_network_ddc(network_name, "depressed")

            # Get control values
            # avg_ctrl = self.get_mean_ddc("control")
            if median==0:
                avg_ctrl = np.nanmean(self.control_weights, axis=0)
            else:
                avg_ctrl = np.nanmedian(self.control_weights, axis=0)

            ctrl_fc = self.subset_fc(avg_ctrl, indices)
            cbar_min = min(ctrl_fc.flatten())
            cbar_max = max(ctrl_fc.flatten())

            # Get depressed values
            # avg_depr = self.get_mean_ddc("depressed")
            if median==0:
                avg_depr = np.nanmean(self.depress_weights, axis=0)
            else:
                avg_depr = np.nanmedian(self.depress_weights, axis=0)

            depr_fc = self.subset_fc(avg_depr, indices)
            cbar_min = min(cbar_min, min(depr_fc.flatten()))
            cbar_max = max(cbar_max, max(depr_fc.flatten()))

            # Plot control
            plt.subplot(131)
            plt.title("{} {} control".format(network_name, hand))
            im = plt.imshow(ctrl_fc, cmap="RdBu_r")
            plt.colorbar()
            # Add gray boxes for self-connections
            for i in range(len(ctrl_fc)):
                im.axes.add_patch(
                    plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
                )
            plt.clim([cbar_min, cbar_max])
            plt.yticks(np.arange(len(indices)), labels)
            plt.xticks(np.arange(len(indices)), labels, rotation="vertical")

            # Plot depressed
            plt.subplot(132)
            plt.title("{} {} depressed".format(network_name, hand))
            im = plt.imshow(depr_fc, cmap="RdBu_r")
            plt.colorbar()
            # Add gray boxes for self-connections
            for i in range(len(depr_fc)):
                im.axes.add_patch(
                    plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
                )
            plt.clim([cbar_min, cbar_max])
            plt.yticks(np.arange(len(indices)), labels)
            plt.xticks(np.arange(len(indices)), labels, rotation="vertical")

            # non parametric statistical test for independent variables
            # _, p = mannwhitneyu(network_ctrl, network_depr)
            from scipy.stats import ttest_ind

            c = network_ctrl.reshape(len(network_ctrl), -1)
            d = network_depr.reshape(len(network_depr), -1)

            t_statistics = np.zeros((np.shape(c)[1], 1))
            p_values = np.zeros((np.shape(c)[1], 1))
            for i in range(np.shape(c)[1]):
                if ttest:
                    t_statistics[i], p_values[i] = ttest_ind(c[:, i], d[:, i])
                else:
                    t_statistics[i], p_values[i] = mannwhitneyu(c[:, i], d[:, i])

            p = p_values.reshape(np.shape(network_ctrl)[1:])

            diff= abs(depr_fc) - abs(ctrl_fc)

            if bonferroni:
                n_comp = network_ctrl.shape[1]
                adjusted_alpha = 0.05 / n_comp
                diff[np.where(p > adjusted_alpha)] = 0
            else:
                diff[np.where(p > 0.05)] = 0

            # Plot if there are significant differences
            # if sum(sum(diff)) > 0:
            plt.subplot(133)
            im = plt.imshow(diff, cmap="RdBu_r")
            # if network_name == "CEN":
            #     plt.clim([-0.0002, 0.0002])
            # else:
            plt.clim([-np.max(diff), np.max(diff)])

            plt.colorbar()

            for i in range(len(diff)):
                im.axes.add_patch(
                    plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
                )
                            
            for i in range(len(ctrl_fc)):
                for j in range(len(ctrl_fc)):
                    if diff[i,j]!=0:
                        if i!=j:
                            if np.sign(ctrl_fc[i,j]) != np.sign(depr_fc[i,j]):
                                plt.scatter(j, i, marker='*', color='k', s=50) 
                                # plt.xlim([0,len(ctrl_fc)])
                                # plt.ylim([0,len(ctrl_fc)])




            # plt.colorbar()
            plt.yticks(np.arange(len(indices)), labels)
            plt.xticks(np.arange(len(indices)), labels, rotation="vertical")
            plt.title("Statisticaly different fc")



            if save_as is not None:
                plt.savefig(f"{self.fig_dir}{save_as}")
            else:
                if bonferroni:
                    plt.savefig(
                        f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix"
                        + str(network_name)
                        + "Bonferroni.svg",
                        format="svg",
                    )
                else:
                    plt.savefig(
                        f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix"
                        + str(network_name)
                        + ".svg",
                        format="svg",
                    )
            # return diff
                    
    def plot_significant_hand_diff_distribution(self, stat_diff, control_R, control_L, ttest=1):

        a = np.where(stat_diff == 1)
        
        # Filter ROIs based on the condition
        valid_rois = [roi for roi in self.all_ROIs if roi not in ['CSF', '3V', '4V']]
        
        # Filter significant differences based on valid ROIs
        valid_diff_indices = [(x, y) for x, y in zip(a[0], a[1]) if self.all_ROIs[x] in valid_rois and self.all_ROIs[y] in valid_rois]
        
        # Calculate the number of subplots based on the number of valid significant differences
        num_plots = len(valid_diff_indices)
        num_cols = 4  # You can adjust the number of columns in the grid
        num_rows = (num_plots + num_cols - 1) // num_cols

        # Create a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))

        for i, (x, y) in enumerate(valid_diff_indices):
            ax = axes.flatten()[i]

            # Create histograms
            sns.histplot(control_R[:, x, y], kde=True, color='green', label='Right', log_scale=(True, False), ax=ax)
            sns.histplot(control_L[:, x, y], kde=True, color='orange', label='Left', log_scale=(True, False), ax=ax)
            ax.set_title(f"{self.all_ROIs[x]}:{self.all_ROIs[y]}")
            ax.legend() 

            # Perform t-test
            if ttest:
                t_stat, p_value = ttest_ind(control_R[:, x, y], control_L[:, x, y])
            else:
                t_stat, p_value = mannwhitneyu(control_R[:, x, y], control_L[:, x, y])

            ax.annotate(f'p-value: {p_value:.4f}', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        return a
    

    def plot_network_connectivity_graph_hand(self, control_R, state, hand, median=0):
        """plot network graph on brain template of connections for left or right handed"""

        coord_list = np.asarray(self.positions[["x", "y", "z"]])
        if median==0:
            R = np.nanmean(control_R, axis=0)
        else:
            R = np.nanmedian(control_R, axis=0)
        if hand=='R':
            color='green'
        else:
            color='orange'
        fig1=plt.figure()
        display = plotting.plot_connectome(
            R,
            coord_list,
            node_color=color,
            edge_cmap=None,
            edge_threshold='95%',
            figure=fig1,
            #             edge_vmax=ev,
            title="{} Right".format(state),
            colorbar=True,
        )

        plotting.show()

        display.savefig(
            f"{self.fig_dir}{self.weights_file_name.split('*')[0]}{hand}_network_connectivity"
            + str(state)
            + ".svg"
        )


    def plot_means_std_matrice_hand(self, control_R, control_L, state, save_as=None, cmap="Reds", median=0):

        if median==0:
            R = np.nanmean(control_R, axis=0)
            L = np.nanmean(control_L, axis=0)
        else:
            R = np.nanmedian(control_R, axis=0)
            L = np.nanmedian(control_L, axis=0)

        R_std=np.nanstd(control_R, axis=0)
        L_std=np.nanstd(control_L, axis=0)

        plt.figure(figsize=(10, 10))

        plt.subplot(221)
        im = plt.imshow(R, cmap="RdBu_r")
        for i in range(len(R)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        # Get R values
        cbar_min = min(R.flatten())
        cbar_max = max(R.flatten())

        # Get L values
        cbar_min = min(cbar_min, min(L.flatten()))
        cbar_max = max(cbar_max, max(L.flatten()))

        plt.clim([cbar_min, cbar_max])
        # plt.clim([-400, 400])
        plt.colorbar()
        plt.title("avg DDC R")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        plt.subplot(222)
        im = plt.imshow(L, cmap="RdBu_r")
        for i in range(len(L)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([cbar_min, cbar_max])
        plt.title("avg DDC L")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.colorbar()

        plt.subplot(223)
        
        im = plt.imshow(R_std, cmap=cmap)
        plt.colorbar()
        # plt.clim([0, 10000])
        plt.title("std DDC R")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        

        # Get control values
        cbar_min = min(R_std.flatten())
        cbar_max = max(R_std.flatten())
        # Get depressed values
        cbar_min = min(cbar_min, min(L_std.flatten()))
        cbar_max = max(cbar_max, max(L_std.flatten()))
        plt.clim([cbar_min, cbar_max])

        plt.subplot(224)
        im = plt.imshow(L_std, cmap=cmap)
        plt.colorbar()
        plt.clim([cbar_min, cbar_max])
        # plt.clim([0, 10000])
        plt.title("std DDC depr")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        if save_as is None:
            plt.savefig(
                f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_RL_mean_std.svg",
                format="svg",
            )
        else:
            plt.savefig(f"{self.fig_dir}{save_as}")


    def plot_interactive_connectivity_graph_hand(self, control_R, state, save_as=None,):

        coord_list = np.asarray(self.positions[["x", "y", "z"]])

        view = plotting.view_connectome(
            control_R,
            coord_list,
            edge_cmap="RdBu_r",
            edge_threshold='95%',
            # symmetric_cmap=False,
            colorbar=True,
            node_color="k",
        )
        return view
    
    def find_outliers_group(self, thr=5):
        from scipy.stats import zscore
        # Assuming your population is stored in a list named 'population'
        # 'population' should contain 1000 20x20 matrices

        # Flatten each matrix into a 1D array
        flattened_population = [individual.flatten() for individual in self.control_weights]

        # Convert the list of flattened arrays into a 2D NumPy array
        data = np.array(flattened_population)

        # Calculate z-scores for each element in the array
        z_scores = zscore(data, axis=None)

        # Define a threshold for outliers (e.g., 3 standard deviations)
        threshold = thr

        # Find indices of outliers
        outlier_indices_C = np.where(np.abs(z_scores) > threshold)

        # Print the indices of outliers
        print("Indices of outliers control:", np.unique(outlier_indices_C[0]))
        
        # Flatten each matrix into a 1D array
        flattened_population = [individual.flatten() for individual in self.depress_weights]

        # Convert the list of flattened arrays into a 2D NumPy array
        data = np.array(flattened_population)

        # Calculate z-scores for each element in the array
        z_scores = zscore(data, axis=None)

        # Define a threshold for outliers (e.g., 3 standard deviations)
        threshold = thr

        # Find indices of outliers
        outlier_indices_D = np.where(np.abs(z_scores) > threshold)

        # Print the indices of outliers
        print("Indices of outliers depressed:", np.unique(outlier_indices_D[0]))
        
        return outlier_indices_C, outlier_indices_D
        
    def dendrogram_DDC_matrices(self, thr=100, n=50):
        print(np.shape(self.control_weights)[0])
        random_numbers = np.random.randint(0, np.shape(self.control_weights)[0], size=n)
        C=np.reshape(self.control_weights, (np.shape(self.control_weights)[0], self.n_roi*self.n_roi))[random_numbers,:]
        random_numbers = np.random.randint(0, np.shape(self.depress_weights)[0], size=n)
        D=np.reshape(self.depress_weights, (np.shape(self.depress_weights)[0], self.n_roi*self.n_roi))[random_numbers,:]
        x=np.vstack([C, D])
        
        
        import scipy.cluster.hierarchy as shc
        dendrogram = shc.linkage(x, method='ward',optimal_ordering=True)
        ordered_indices = shc.leaves_list(dendrogram)

        # corr_matrix_reordered = corr_vec[ordered_indices, :]
        # corr_matrix_reordered = corr_matrix_reordered[:, ordered_indices]
        # Plot the dendrogram
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(9, 5), constrained_layout=True)
        plt.title("Dendrogram")
        # # plt.subplot(311)
        R=shc.dendrogram(dendrogram,color_threshold=thr)
        
        return C, D, x, R, ordered_indices
        
    def check_stats(self):
        from scipy.stats import ttest_ind
        from scipy.stats import mannwhitneyu
        from scipy.stats import shapiro
        
        t_statistics = np.zeros((np.shape(self.control)[1], 1))
        p_values_t = np.zeros((np.shape(self.control)[1], 1))
        p_values_t_log = np.zeros((np.shape(self.control)[1], 1))
        p_values_mw = np.zeros((np.shape(self.control)[1], 1))
        p_value_norm_c = np.zeros((np.shape(self.control)[1], 1))
        p_value_norm_d = np.zeros((np.shape(self.control)[1], 1))

        for i in range(np.shape(self.control)[1]):
            t_statistics[i], p_values_t[i] = ttest_ind(
                self.control[:, i], self.depress[:, i]
            )
            t_statistics[i], p_values_t_log[i] = ttest_ind(
                np.log(np.abs(self.control[:, i])), np.log(np.abs(self.depress[:, i]))
            )
            t_statistics[i], p_values_mw[i] = mannwhitneyu(self.control[:, i], self.depress[:, i])
            stat, p_value_norm_c[i] = shapiro(self.control[:, i])
            stat, p_value_norm_d[i] = shapiro(self.depress[:, i])
            
        return p_values_t, p_values_t_log, p_values_mw, p_value_norm_c,p_value_norm_d, self.control, self.depress
        