from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

class Analyze_DDC:
    def __init__(
        self,
        dataset_1,
        dataset_2,
        dataset_names=["dataset_1", "dataset_2"],
        ind=True,  # if the datasets are independent
        verbose=False,
    ):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.ind = ind
        self.pvalue_threshold = 0.05
        self.verbose = verbose
        self.dataset_names = dataset_names

    def is_normal(self):
        # Check if the data is normally distributed (shapiro-wilk)
        if len(self.dataset_1) < 5000:
            _, p = stats.shapiro(self.dataset_1)
            if p < self.pvalue_threshold:
                is_normal = False
                if self.verbose:
                    print("Dataset 1 is not normally distributed.")
            else:
                _, p = stats.shapiro(self.dataset_2)
                if p < self.pvalue_threshold:
                    is_normal = False
                    if self.verbose:
                        print("Dataset 2 is not normally distributed.")

                # both distributions need to be normally distributed
                else:
                    is_normal = True
                    if self.verbose:
                        print("Both distributions are normally distributed.")
        else:
            is_normal = True

        return is_normal

    def compare(self, plot=False):
        if self.verbose:
            print("STARTING ANALYSIS")

        n_rois_sqrd = self.dataset_1.shape[1]
        n_rois = int(np.sqrt(n_rois_sqrd))

        sig_connections = np.zeros(self.dataset_1.shape[1])

        is_normal = self.is_normal()
        if is_normal and self.ind:
            if self.verbose:
                print("Data is normally distributed and independent.")
                print("Using independent t-test to compare datasets.")
            _, p = stats.ttest_ind(self.dataset_1, self.dataset_2)
        elif is_normal and not self.ind:
            if self.verbose:
                print("Data is normally distributed and dependent.")
                print("Using dependent t-test to compare datasets.")
            _, p = stats.ttest_rel(self.dataset_1, self.dataset_2)
        elif not is_normal and self.ind:
            if self.verbose:
                print("Data is independent but not normally distributed.")
                print("Using Mann-Whitney U test to compare datasets.")
            _, p = stats.mannwhitneyu(self.dataset_1, self.dataset_2)
        elif not is_normal and not self.ind:
            if self.verbose:
                print("Data is dependent but not normally distributed.")
                print("Using Wilcoxon test to compare datasets.")
            _, p = stats.wilcoxon(self.dataset_1, self.dataset_2)

        # get matrix of significant connections
        sig_connections[np.where(p < self.pvalue_threshold)[0]] = 1
        sig_connections = np.reshape(sig_connections, (n_rois, n_rois))
        
        if plot:
            plt.figure()
            plt.imshow(sig_connections)
            plt.show()

        return sig_connections

