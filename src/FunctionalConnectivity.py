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
                "/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/baseline/downloads/sub-NDARINV04GAB2AA/ROIs_centroid_coordinates.csv"
            )
        else:
            self.positions = pd.read_csv(
                "/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/baseline/downloads/sub-NDARINV04GAB2AA/ROIs_centroid_coordinates.csv"
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
                    if self.labels.values[i] == 0:
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

    def scale_inter_sub(self):
        return scaled_derp, scaled_controls

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
        self, colorbar=True, save_as=None, bonferroni=False
    ):
        # s, p = mannwhitneyu(self.control, self.depress)
        stat_diff = np.zeros(self.n_roi * self.n_roi)

        from scipy.stats import ttest_ind

        t_statistics = np.zeros((np.shape(self.control)[1], 1))
        p_values = np.zeros((np.shape(self.control)[1], 1))

        for i in range(np.shape(self.control)[1]):
            t_statistics[i], p_values[i] = ttest_ind(
                self.control[:, i], self.depress[:, i]
            )
            # t_statistics[i], p_values[i] = mannwhitneyu(c[:, i], d[:, i])

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
    
    def plot_significant_connections_bar(self, bonferroni=False):

        stat_diff = np.zeros(self.n_roi * self.n_roi)

        from scipy.stats import ttest_ind

        t_statistics = np.zeros((np.shape(self.control)[1], 1))
        p_values = np.zeros((np.shape(self.control)[1], 1))

        for i in range(np.shape(self.control)[1]):
            t_statistics[i], p_values[i] = ttest_ind(
                self.control[:, i], self.depress[:, i]
            )
            # t_statistics[i], p_values[i] = mannwhitneyu(c[:, i], d[:, i])

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

    def plot_means_std_matrices(self, save_as=None,cmap="Reds"):
        avg_ddc_ctrl = self.get_mean_ddc("control")
        avg_ddc_depr = self.get_mean_ddc("depressed")
        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        im = plt.imshow(avg_ddc_ctrl, cmap="RdBu_r")
        for i in range(len(avg_ddc_ctrl)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([-400, 400])
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
        plt.clim([-400, 400])
        plt.title("avg DDC depr")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.colorbar()

        plt.subplot(223)
        std_ddc_ctrl = self.get_std_ddc("control")
        im = plt.imshow(std_ddc_ctrl, cmap=cmap)
        plt.colorbar()
        plt.clim([0, 10000])
        plt.title("std DDC control")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.subplot(224)
        std_ddc_depr = self.get_std_ddc("depressed")
        im = plt.imshow(std_ddc_depr, cmap=cmap)
        plt.colorbar()
        plt.clim([0, 10000])
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
            plt.title("Subject: " + n)

            plt.axis("off")
            k = k + 1

    def subset_fc(self, fc, include):
        """get subnetwork connectivity"""
        a = fc[include, :]
        a = a[:, include]
        return a

    def plot_network_heatmap(self, network_name, save_as=None, bonferroni=False):
        """plot binary matrix of significant connections for a specific subnetwork"""
        plt.figure(figsize=(15, 5))
        scaled_c, scaled_d = self.standard_scaling()

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
        avg_ctrl = np.mean(self.control_weights, axis=0)
        ctrl_fc = self.subset_fc(avg_ctrl, indices)
        cbar_min = min(ctrl_fc.flatten())
        cbar_max = max(ctrl_fc.flatten())

        # Get depressed values
        # avg_depr = self.get_mean_ddc("depressed")
        avg_depr = np.mean(self.depress_weights, axis=0)
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
            t_statistics[i], p_values[i] = ttest_ind(c[:, i], d[:, i])
            # t_statistics[i], p_values[i] = mannwhitneyu(c[:, i], d[:, i])

        p = p_values.reshape(np.shape(network_ctrl)[1:])
        # _, p = ttest_ind(network_ctrl, network_depr)
        # diff = depr_fc - ctrl_fc
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
        self, network_name, ev, save_as=None, bonferroni=False
    ):
        """plot differences between ctrl and depressed network graph on brain template for a specific subnetwork"""
        scaled_c, scaled_d = self.standard_scaling()

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

        # non parametric statistical test for independent variables
        # _, p = mannwhitneyu(network_ctrl, network_depr)

        from scipy.stats import ttest_ind

        _, p = ttest_ind(network_ctrl, network_depr)
        diff = abs(depr_fc) - abs(ctrl_fc)

        if bonferroni:
            n_comp = network_ctrl.shape[1]
            adjusted_alpha = 0.05 / n_comp
            diff[np.where(p > adjusted_alpha)] = 0
        else:
            diff[np.where(p > 0.05)] = 0

        #         if sum(sum(diff)) > 0:
        #             if ev == []:
        #                 ev = 0.002

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

    def plot_connection_probDistr(self, x=None, y=None, save_as=None):
        "plotting the distribution for the selected connection"
        "x and y can be int = index of the desired ROIs or str + name of the desired ROIs to compare"
        import plotly.figure_factory as ff

        scaled_c, scaled_d = self.standard_scaling()

        if isinstance(x, str):
            x = self.all_ROIs.index(x)
            y = self.all_ROIs.index(y)

        hist_data = [scaled_c[:, x, y], scaled_d[:, x, y]]
        group_labels = ["Control", "Depressed"]
        plt.figure(figsize=(10, 10))
        fig = ff.create_distplot(hist_data, group_labels, show_hist=True)
        # Add title
        title = f"{self.all_ROIs[x]}:{self.all_ROIs[y]}"
        fig.update_layout(title_text=title)
        fig.show()

        if save_as is None:
            plt.savefig(
                f"{self.fig_dir}{self.all_ROIs[x]}{self.all_ROIs[y]}_distributions"
                + ".svg"
            )
        else:
            plt.savefig(f"{self.fig_dir}{save_as}")
