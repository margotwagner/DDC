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

        # builds the dataset
        dataset = self.build_dataset()
        # assign the output to the variables
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
        ) = dataset

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
        j = 0
        k = 0

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
                                control_weights[j, :, :] = np.asarray(
                                    pd.read_csv(f, header=None)
                                )

                                # Threshold binarization (to be replaced by bootstrap) and reshape
                                control_weights_vec[j, :] = np.reshape(
                                    (abs(control_weights[j, :, :]) > self.thrs) * 1,
                                    (1, self.n_roi * self.n_roi),
                                )

                                ctrl_files.append(f)

                                j += 1

                    # Depressed subjects
                    else:
                        d = np.asarray(pd.read_csv(f, header=None))
                        if not len(d) < self.n_roi:
                            if sum(sum(np.isnan(d))) < 1:
                                depress_weights[k, :, :] = np.asarray(
                                    pd.read_csv(f, header=None)
                                )
                                # Threshold binarization (to be replaced by bootstrap) and reshape
                                depress_weights_vec[k, :] = np.reshape(
                                    (abs(depress_weights[k, :, :]) > self.thrs) * 1,
                                    (1, self.n_roi * self.n_roi),
                                )

                                k += 1

                                depr_files.append(f)

                else:
                    no_weights.append(f)

        control_weights = control_weights[:j, :, :]
        depress_weights = depress_weights[:k, :, :]
        control_weights_vec = control_weights_vec[:j, :]
        depress_weights_vec = depress_weights_vec[:k, :]
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
        plt.imshow(avg, vmin=-limit, vmax=limit)
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
        avg_diff = avg_ctrl - avg_depr
        limit = max(abs(np.min(avg_diff)), abs(np.max(avg_diff)))
        plt.imshow(avg_diff, vmin=-limit, vmax=limit)
        # plt.clim([-0.01, 0.01])
        plt.title("Difference")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

    def plot_significant_connections_matrix(self, colorbar=False, save_as=None):
        s, p = mannwhitneyu(self.control, self.depress)
        stat_diff = np.zeros(self.n_roi * self.n_roi)
        stat_diff[np.where(p < 0.05)[0]] = 1
        stat_diff = np.reshape(stat_diff, (self.n_roi, self.n_roi))
        a = self.get_mean_ddc("control")
        b = self.get_mean_ddc("depressed")
        diff = abs(b) - abs(a)
        diff[np.where(stat_diff == 0)] = 0

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(stat_diff)
        plt.title("Statistically different connections")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.subplot(122)
        im = plt.imshow(diff, cmap="RdBu")
        for i in range(len(diff)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([-np.max(diff), np.max(diff)])
        plt.colorbar()
        plt.title("Statistically different connections")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        p_table = pd.DataFrame(columns=["Areas", "p-value"])
        pv = np.reshape(p, (self.n_roi, self.n_roi))
        for i in range(self.n_roi):
            for j in range(self.n_roi):
                new_row = {
                    "Areas": self.all_ROIs[i] + "-" + self.all_ROIs[j],
                    "p-value": pv[i, j],
                }
                p_table = p_table.append(new_row, ignore_index=True)

        # p_table.to_csv("/home/acamassa/ABCD/DDC/figures/p_values_table.csv")

        if save_as is None:
            plt.savefig(
                f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix.svg",
                format="svg",
            )
        else:
            plt.savefig(f"{self.fig_dir}{save_as}")

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

    def plot_means_std_matrices(self, save_as=None):
        avg_ddc_ctrl = self.get_mean_ddc("control")
        avg_ddc_depr = self.get_mean_ddc("depressed")
        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        im = plt.imshow(avg_ddc_ctrl)
        for i in range(len(avg_ddc_ctrl)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        # plt.clim([-0.01, 0.01])
        plt.title("avg DDC control")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        plt.subplot(222)
        im = plt.imshow(avg_ddc_depr)
        for i in range(len(avg_ddc_depr)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        # plt.clim([-0.01, 0.01])
        plt.title("avg DDC depr")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        # plt.colorbar()

        plt.subplot(223)
        std_ddc_ctrl = self.get_std_ddc("control")
        im = plt.imshow(std_ddc_ctrl)
        # plt.clim([0, 1])
        plt.title("std DDC control")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.subplot(224)
        std_ddc_depr = self.get_std_ddc("depressed")
        im = plt.imshow(std_ddc_depr)
        # plt.clim([0, 1])
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
            plt.imshow(DDC[i, :, :])
            # plt.clim([-0.1, 0.1])
            plt.colorbar()

            plt.axis("off")
            k = k + 1

    def subset_fc(self, fc, include):
        """get subnetwork connectivity"""
        a = fc[include, :]
        a = a[:, include]
        return a

    def plot_network_heatmap(self, network_name, save_as=None):
        """plot binary matrix of significant connections for a specific subnetwork"""
        plt.figure(figsize=(15, 5))

        indices = self.get_network_indices(network_name)
        labels = self.get_network_labels(network_name)
        network_ctrl = self.get_network_ddc(network_name, "control")
        network_depr = self.get_network_ddc(network_name, "depressed")

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

        # Plot control
        plt.subplot(131)
        plt.title("{} control".format(network_name))
        im = plt.imshow(ctrl_fc, cmap="RdBu_r")
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
        # Add gray boxes for self-connections
        for i in range(len(depr_fc)):
            im.axes.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
            )
        plt.clim([cbar_min, cbar_max])
        plt.yticks(np.arange(len(indices)), labels)
        plt.xticks(np.arange(len(indices)), labels, rotation="vertical")

        # non parametric statistical test for independent variables
        _, p = mannwhitneyu(network_ctrl, network_depr)

        # Plot difference
        diff = abs(depr_fc) - abs(ctrl_fc)
        diff[np.where(p > 0.05)] = 0  # mask non-significant connections
        # Plot if there are significant differences
        if sum(sum(diff)) > 0:
            plt.subplot(133)
            im = plt.imshow(diff, cmap="RdBu_r")
            for i in range(len(diff)):
                im.axes.add_patch(
                    plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="gray")
                )
            plt.clim([-np.max(diff), np.max(diff)])
            # plt.colorbar()
            plt.yticks(np.arange(len(indices)), labels)
            plt.xticks(np.arange(len(indices)), labels, rotation="vertical")
            plt.title("Statisticaly different fc")

        if save_as is not None:
            plt.savefig(f"{self.fig_dir}{save_as}")
        else:
            plt.savefig(
                f"{self.fig_dir}{self.weights_file_name.split('*')[0]}_sig_conn_matrix"
                + str(network_name)
                + ".svg",
                format="svg",
            )

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

    def plot_network_connectivity_graph_diff(self, network_name, ev, save_as=None):
        """plot differences between ctrl and depressed network graph on brain template for a specific subnetwork"""
        indices = self.get_network_indices(network_name)
        coord_list = np.asarray(self.positions[["x", "y", "z"]])

        # non parametric statistical test for independent variables
        network_control = self.get_network_ddc(network_name, "control")
        network_depr = self.get_network_ddc(network_name, "depressed")
        _, p = mannwhitneyu(network_control, network_depr)

        avg = self.get_mean_ddc("control")
        a = self.subset_fc(avg, indices)
        avg = self.get_mean_ddc("depressed")
        b = self.subset_fc(avg, indices)
        diff = abs(b) - abs(a)
        diff[np.where(p > 0.05)] = 0
        if sum(sum(diff)) > 0:
            if ev == []:
                ev = 0.002

            display = plotting.plot_connectome(
                diff,
                coord_list[indices, :],
                node_color="k",
                edge_cmap="RdBu",
                edge_vmax=ev,
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
        else:
            print("no different connections to plot")

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
        diff = abs(b) - abs(a)
        diff[np.where(stat_diff == 0)] = 0

        coord_list = np.asarray(self.positions[["x", "y", "z"]])

        view = plotting.view_connectome(
            diff,
            coord_list,
            edge_cmap="RdBu",
            edge_threshold=50,
            symmetric_cmap=False,
            colorbar=True,
            node_color="k",
        )
        # pd.DataFrame(diff).to_csv('/home/acamassa/ABCD/DDC/figures/stat_diff_matrix.csv')
        return view
