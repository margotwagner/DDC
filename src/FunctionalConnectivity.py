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
        fig_dir,
    ):
        self.labels = labels
        self.n_roi = n_roi
        self.thrs = thrs
        self.weights_file_name = weights_file_name
        self.fig_dir = fig_dir #"../figures/"

        # TODO: specify from data rather than hardcoding
        self.n1 = 4669 
        self.n2 = 1056
        
        self.DDC_path = (
            "/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/DDC/baseline/raw/"
        )
            
        self.DMN_indices = [12, 21, 6, 23, 31, 46, 55, 40, 57, 65]
        self.DMN_labels = [
            "l-mOFC",
            "l-PCC",
            "l-IPL",
            "l-Prec",
            "l-TPO",
            "r-mOFC",
            "r-PCC",
            "r-IPL",
            "r-Prec",
            "r-TPO",
        ]
        if self.weights_file_name.startswith("subc_"):
            y=30
            self.DMN_indices=list(map(y.__add__, self.DMN_indices))
            
        self.CEN_indices = [6, 12, 25, 26, 27, 40, 46, 59, 60, 61]
        self.CEN_labels = [
            "l-IPL",
            "l-mOFC",
            "l-mPF",
            "l-SPG",
            "l-SPL",
            "r-IPL",
            "r-mOFC",
            "r-mPF",
            "r-SPG",
            "r-SPL",
        ]
        if self.weights_file_name.startswith("subc_"):
            y=30
            self.CEN_indices=list(map(y.__add__, self.CEN_indices))
            

        if self.weights_file_name.startswith("subc_"):
            self.SN_indices = [13, 63, 54, 61, 27, 97, 88, 95]
            self.SN_labels = [
                "l-Amy",
                "l-Ins",
                "l-raACC",
                "l-TP",
                "r-Amy",
                "r-Ins",
                "r-raACC",
                "r-TP",
                
            ]
        else:
            self.SN_indices = [33, 24, 31, 67, 58, 65]
            self.SN_labels = [
                "l-Ins",
                "l-raACC",
                "l-TP",
                "r-Ins",
                "r-raACC",
                "r-TP",
            ]
            
        self.positions = pd.read_csv(
            "/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/baseline/downloads/sub-NDARINV04GAB2AA/ROIs_centroid_coordinates.csv"
        )

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
                    try:
                        # Control subjects
                        if self.labels.values[i] == 0:
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

                    except:
                        # no DDC ROIs available
                        missing_rois.append(f)
                else:
                    no_weights.append(f)

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

        plt.savefig(f"{self.fig_dir}sig_conn_bar.eps", format="eps")


    def plot_significant_connections_graph(self):
        """plot network graph of signifiicant connections for controls and depressed"""
        sig_connect_control = self.get_binary_connections_percentage_control()
        sig_connect_depress = self.get_binary_connections_percentage_control()
        
        if self.weights_file_name.startswith("subc_"):
            thr=0.25
        else:
            thr=0.65
            
        ddc_plot = sig_connect_control > thr
        G = nx.from_numpy_array(ddc_plot)
        plt.figure(figsize=(10, 10))
        if not self.weights_file_name.startswith("subc_"):
            self.positions=self.positions[30:]
        plt.subplot(121)
        nx.draw(G, np.asarray(self.positions[["x", "y"]]), with_labels=True)
        plt.title("Control")
        plt.subplot(122)
        ddc_plot = sig_connect_depress > thr
        G = nx.from_numpy_array(ddc_plot)
        plt.title("Depressed")
        nx.draw(G, np.asarray(self.positions[["x", "y"]]), with_labels=True)

        plt.savefig(f"{self.fig_dir}sig_conn_graph.eps", format="eps")


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

    def plot_significant_connections_matrix(self, colorbar=False):
        s, p = mannwhitneyu(self.control, self.depress)
        stat_diff = np.zeros(self.n_roi * self.n_roi)
        stat_diff[np.where(p < 0.05)[0]] = 1
        stat_diff = np.reshape(stat_diff, (self.n_roi, self.n_roi))
        plt.figure()
        plt.imshow(stat_diff)
        plt.title("Statistically different connections")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        if not self.weights_file_name.startswith("subc_"):
            self.positions=self.positions[30:]
        plt.figure(figsize=(15, 15)) 

        G = nx.from_numpy_array(stat_diff)
        plt.title("Statistically different connections")
        nx.draw(G, np.asarray(self.positions[["x", "y"]]), with_labels=True)


        plt.savefig(f"{self.fig_dir}sig_conn_matrix.eps", format="eps")


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

    def plot_means_std_matrices(self):
        avg_ddc_ctrl = self.get_mean_ddc("control")
        avg_ddc_depr = self.get_mean_ddc("depressed")
        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        im = plt.imshow(avg_ddc_ctrl)
        # plt.clim([-0.01, 0.01])
        plt.title("avg DDC control")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        plt.subplot(222)
        im = plt.imshow(avg_ddc_depr)
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
        
    def plot_random_matrices(self, state):
        """plot 25 random DDC matrices"""
        if state=="control":
            DDC = self.control_weights
        else:
            DDC = self.depress_weights
        n=random.sample(range(len(DDC)),25)
        fig=plt.figure(figsize=(10,10))
        k=0
        for i in n:
            ax = fig.add_subplot(5,5,k+1)
            plt.imshow(DDC[i,:,:])
            plt.clim([-0.1,0.1])

            plt.axis('off')
            k=k+1

    def subset_fc(self, fc, include):
        """get subnetwork connectivity"""
        a = fc[include, :]
        a = a[:, include]
        return a

    def plot_network_condition(
        self, network_indices, network_labels, network_name, state, plot, subplot_number
    ):
        """plot connectivity matrix for a specific subnetwork"""
        plt.subplot(subplot_number)

        if state == "control":
            plt.title("{} control".format(network_name))
        else:
            plt.title("{} depressed".format(network_name))

        avg = self.get_mean_ddc(state)

        a = self.subset_fc(avg, network_indices)

        plt.imshow(a)
        plt.yticks(np.arange(len(network_indices)), network_labels)
        plt.xticks(np.arange(len(network_indices)), network_labels, rotation="vertical")

    def plot_network_heatmap(self, network_name):
        """plot binary matrix of significant connections for a specific subnetwork"""
        plt.figure(figsize=(15, 5))

        indices = self.get_network_indices(network_name)
        labels = self.get_network_labels(network_name)
        network_control = self.get_network_ddc(network_name, "control")
        network_depr = self.get_network_ddc(network_name, "depressed")

        # Plot control
        self.plot_network_condition(indices, labels, network_name, "control", plt, 131)

        # Plot depressed
        self.plot_network_condition(
            indices, labels, network_name, "depressed", plt, 132
        )

        # non parametric statistical test for independent variables
        _, p = mannwhitneyu(network_control, network_depr)

        plt.subplot(133)
        plt.imshow(p < 0.05)
        plt.yticks(np.arange(len(indices)), labels)
        plt.xticks(np.arange(len(indices)), labels, rotation="vertical")
        plt.title("Statisticaly different fc")

        # plt.title('DMN abs difference')
        # plt.imshow(abs(a-a_d))
        # plt.yticks(np.arange(len(DMN)),labels)
        # plt.xticks(np.arange(len(DMN)),labels, rotation='vertical')
        # for r in DMN:
        #    print(Desikan_ROIs[r])

    def plot_network_siignificant_connections_graph(self, network_name):
        """plot network graph only significantly different connections for a specific subnetwork"""
        indices = self.get_network_indices(network_name)
        labels = self.get_network_labels(network_name)

        # non parametric statistical test for independent variables
        network_control = self.get_network_ddc(network_name, "control")
        network_depr = self.get_network_ddc(network_name, "depressed")
        _, p = mannwhitneyu(network_control, network_depr)

        G = nx.from_numpy_array(p < 0.05)
        if not self.weights_file_name.startswith("subc_"):
            self.positions=self.positions[30:]
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
            f"{self.fig_dir}sig_conn_network" + str(network_name) + ".eps",
            format="eps",
        )


    def plot_network_connectivity_graph(self, network_name, state):
        """plot network graph on brain template of connections for a specific subnetwork and state"""
        indices = self.get_network_indices(network_name)
        if not self.weights_file_name.startswith("subc_"):
            self.positions=self.positions[30:]
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
            edge_vmax=ev,
            title="{} {}".format(state, network_name),
            colorbar=True,
        )

        plotting.show()

        display.savefig(
            f"{self.fig_dir}network_connectivity"
            + str(network_name)
            + str(state)
            + ".eps"
        )


    def plot_connectivity_graph(self, state):
        """plot network graph on brain template of all connections for a specific state"""
        if not self.weights_file_name.startswith("subc_"):
            self.positions=self.positions[30:]
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
            f"{self.fig_dir}network_connectivity" + str(state) + ".eps"
        )


    def plot_interactive_connectivity_graph(self, state):
        """interactive 3D plot network graph on brain template of all connections for a specific state"""
        if not self.weights_file_name.startswith("subc_"):
            self.positions=self.positions[30:]
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
