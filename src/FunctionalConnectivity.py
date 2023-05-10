# Class for DDC dataset
# TODO: generalize functions so they can take in depressed *or* control
# TODO: better way to handle network name -- assign globally?


# Any import statements go here, above the class declaration
import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import mannwhitneyu
from nilearn import datasets, plotting


# this is where we declare our class
class FunctionalConnectivity:
    # this function initializes the class, so here we initialize the dataset. Every class has an initialization function
    # this is similar to the initialization you did in the notebook, so ill maintain the variable names for now

    # if you want the user to be able to specify something, put it here. Same like a function.
    def __init__(
        self,
        labels,
        n_roi,  # can probably set n_roi from labels, right? like len(labels)?
        weights_file_name,
    ):
        # if you want the variable to be automatically specified, only put it down here

        # for the variables you allow users to specify, you still have to tell
        # the class what they are, so it looks like this
        self.labels = labels
        self.n_roi = n_roi
        self.weights_file_name = weights_file_name

        # these are automatically specified
        self.n1 = (
            1518  # NOT SURE WHERE THIS CAME FROM (n_control - TODO: get automatically)
        )
        self.n2 = 267  # NOT SURE WHERE THIS CAME FROM (n_depressed)
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
        self.CEN_indices = [1, 6, 25, 33, 35, 40, 59, 67]
        self.CEN_labels = [
            "l-caACC",
            "l-IPL",
            "l-rmFG",
            "l-Ins",
            "r-caACC",
            "r-IPL",
            "r-rmFG",
            "r-Ins",
        ]
        self.SN_indices = [33, 23, 1, 16, 30, 67, 57, 35, 50, 64]
        self.SN_labels = [
            "l-Ins",
            "l-Prec",
            "l-caACC",
            "l-Parsop",
            "l-Fpole",
            "r-Ins",
            "r-Prec",
            "r-caACC",
            "r-Parsop",
            "r-Fpole",
        ]

        self.positions = pd.read_csv(
            "/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/baseline/downloads/sub-NDARINV1L5VJRZG/ROIs_centroid_coordinates.csv"
        )[30:]

        # we can also specify it with a function
        # call the function that builds the dataset
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

    # this is how you actually build the dataset, so it should be called when the dataset is initialized
    # self is included automatically in any function in this class (including __init__)
    def build_dataset(self, is_cov=False):
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
        # to use variables, you have to include "self." before the name when calling them (labels -> self.labels)
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

                            # TODO: explain this
                            control_weights_vec[j, :] = np.reshape(
                                (abs(control_weights[j, :, :]) > 0.1) * 1,
                                (1, self.n_roi * self.n_roi),
                            )

                            ctrl_files.append(f)

                            j += 1

                        # Depressed subjects
                        else:
                            depress_weights[k, :, :] = np.asarray(
                                pd.read_csv(f, header=None)
                            )

                            depress_weights_vec[k, :] = np.reshape(
                                (abs(depress_weights[k, :, :]) > 0.1) * 1,
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
        # This isn't the cleanest, but it's fine for now
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

    def get_significant_connections_control(self):
        sig_conn = np.reshape(
            sum(self.control_weights_vec) / np.shape(self.control_weights_vec)[0],
            (self.n_roi, self.n_roi),
        )

        return sig_conn

    def get_significant_connections_depress(self):
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

    def plot_weights(self, state, plot=plt.figure(), colorbar=True):
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

    def plot_weights_across_states(self):
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        self.plot_weights("control", plt)

        plt.subplot(122)
        self.plot_weights("depressed", plt, colorbar=False)

    def plot_2(self):
        sig_connect_control = self.get_significant_connections_control()
        sig_connect_depress = self.get_significant_connections_depress()
        plt.figure()
        plt.title("Significant connection % across Control individuals")
        plt.imshow(sig_connect_control, cmap="Greens")
        plt.colorbar()
        plt.figure()
        plt.title("Significant connection % across Depressed individuals")
        plt.imshow(sig_connect_depress, cmap="Reds")
        plt.colorbar()
        plt.figure()
        plt.title("Significant connection abs difference")
        plt.imshow(abs(sig_connect_control - sig_connect_depress), cmap="Greys")
        plt.colorbar()

    def plot_3(self):
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
        )
        plt.xlabel("Connections#")
        plt.ylabel("counts")
        plt.legend()
        plt.grid()
        plt.ylim([0, 1])
        plt.title("Significant connections (abs(DDC)>0.1)")

    def plot_4(self):
        sig_connect_control = self.get_significant_connections_control()
        sig_connect_depress = self.get_significant_connections_depress()
        ddc_plot = sig_connect_control > 0.8
        G = nx.from_numpy_array(ddc_plot)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        nx.draw(G, np.asarray(self.positions[["x", "y"]]), with_labels=True)
        plt.title("Control")
        plt.subplot(122)
        ddc_plot = sig_connect_depress > 0.8
        G = nx.from_numpy_array(ddc_plot)
        plt.title("Depressed")
        nx.draw(G, np.asarray(self.positions[["x", "y"]]), with_labels=True)

    def plot_5(self):
        fig = plt.figure(figsize=(10, 10))
        n = 0
        for i in range(25):
            ax = fig.add_subplot(5, 5, n + 1)
            plt.imshow(self.control_weights[i, :, :])
            plt.clim([-0.5, 0.5])

            plt.axis("off")
            n += 1

    def plot_mean_weights(self, state, plot=plt.figure(), colorbar=True):
        avg = self.get_mean_ddc(state)
        plt.imshow(avg)
        if colorbar:
            plt.colorbar()
        plt.clim([-0.05, 0.05])
        plt.title("average {} weights".format(state))
        plt.xlabel("ROI #")
        plt.ylabel("ROI #"),

    def plot_mean_difference(self, plot=plt.figure()):
        avg_ctrl = self.get_mean_ddc("control")
        avg_depr = self.get_mean_ddc("depressed")
        plt.imshow(avg_ctrl - avg_depr)
        plt.clim([-0.05, 0.05])
        plt.title("Difference")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

    def plot_means(self, colorbar=False):
        plt.figure(figsize=(10, 5))

        plt.subplot(131)
        self.plot_mean_weights("control", plt, colorbar)

        plt.subplot(132)
        self.plot_mean_weights("depressed", plt, colorbar)

        plt.subplot(133)
        # Rho_diff
        self.plot_mean_difference(plt)

        plt.tight_layout()

        # plt.colorbar()

    def plot_9(self):
        avg_ddc_ctrl = self.get_mean_ddc("control")
        avg_ddc_depr = self.get_mean_ddc("depressed")
        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        im = plt.imshow(avg_ddc_ctrl)
        plt.clim([-0.05, 0.05])
        plt.title("avg DDC control")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        plt.subplot(222)
        im = plt.imshow(avg_ddc_depr)
        plt.clim([-0.05, 0.05])
        plt.title("avg DDC depr")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

        plt.subplot(223)
        std_ddc_ctrl = self.get_std_ddc("control")
        im = plt.imshow(std_ddc_ctrl)
        plt.clim([0, 1])
        plt.title("std DDC control")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")
        plt.subplot(224)
        std_ddc_depr = self.get_std_ddc("depressed")
        im = plt.imshow(std_ddc_depr)
        plt.clim([0, 1])
        plt.title("std DDC depr")
        plt.xlabel("ROI #")
        plt.ylabel("ROI #")

    def subset_fc(self, fc, include):
        a = fc[include, :]
        a = a[:, include]
        return a

    def plot_network_condition(
        self, network_indices, network_labels, network_name, state, plot, subplot_number
    ):
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
        plt.clim([-0.05, 0.05])

    def plot_network_heatmap(self, network_name):
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

    def plot_network(self, network_name):
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
        nx.draw_networkx(
            G,
            coord_list[indices, :],
            node_color="orange",
            with_labels=True,
            labels=labeldict,
        )
        plt.title("{} significantly different connections".format(network_name))

    def plot_network_connectivity(self, network_name, state):
        indices = self.get_network_indices(network_name)

        coord_list = np.asarray(self.positions[["x", "y", "z"]])
        avg = self.get_mean_ddc(state)
        a = self.subset_fc(avg, indices)

        plotting.plot_connectome(
            a,
            coord_list[indices, :],
            node_color="r",
            edge_cmap=None,
            edge_vmax=0.07,
            title="{} {}".format(state, network_name),
            colorbar=True,
        )

        plotting.show()

    def plot_connectivity(self, state):
        coord_list = np.asarray(self.positions[["x", "y", "z"]])

        avg = self.get_mean_ddc(state)
        plotting.plot_connectome(
            avg,
            coord_list,
            edge_cmap=None,
            edge_threshold="98%",
            title="{}".format(state),
        )
        plotting.show()

    def plot_interactive_connectivity(self, state):
        coord_list = np.asarray(self.positions[["x", "y", "z"]])
        avg = self.get_mean_ddc(state)

        view = plotting.view_connectome(
            avg,
            coord_list,
            edge_cmap=None,
            edge_threshold="98%",
            title="{}".format(state),
            symmetric_cmap=False,
        )

        return view

        # uncomment this to open the plot in a web browser:
        # view.open_in_browser()
        # view
