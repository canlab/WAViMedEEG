import os
import numpy as np
import config
import random
import math
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from Prep import Contig, Spectra


class MidpointNormalize(Normalize):
    """
    Utility function to move the midpoint of a colormap to be around
    the values of interest
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):

        self.midpoint = midpoint

        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))


# takes one positional argument: either "contigs" or "spectra"
class Classifier:
    """
    Class object to which we can load our data before differentiating
    using various ML methods

    Parameters:
        - type (required positional): "contigs" or "spectra"
        - network_channels (default): list of channel names to be included
    """

    def __init__(
        self,
        type,
        network_channels=config.network_channels,
            freq_snip=None):

        self.data = []

        self.type = type

        self.network_channels = config.network_channels

        self.trial_name = None

        self.freq_snip = None

    def LoadData(self, path):
        """
        Loads one data at a time, appending it to the Classifier.data attribute

        Parameters:
            - path (required positional): path of file (spectra or contig)
              to be loaded and stacked on the parent object's 'data' attribute
              note: objects will be loaded in as Contig or Spectra objects
        """
        fname = os.path.basename(path)

        self.trial_name = os.path.basename(
            os.path.dirname(
                os.path.dirname(path)))\
            + "/"+os.path.basename(os.path.dirname(path))

        if self.type == "contigs" or self.type == "erps":
            self.data.append(
                Contig(
                    np.genfromtxt(path, delimiter=","),
                    fname.split('_')[2][:-4],
                    fname[:config.participantNumLen],
                    fname.split('_')[1],
                    source=os.path.basename(os.path.dirname(path))))

        elif self.type == "spectra":
            self.data.append(
                Spectra(
                    np.genfromtxt(path, delimiter=",").T[1:].T,
                    np.genfromtxt(path, delimiter=",").T[0].T,
                    fname.split('_')[2][:-4], fname[:config.participantNumLen],
                    fname.split('_')[1],
                    source=os.path.basename(os.path.dirname(path))))

        if (self.data[-1].data.shape != self.data[0].data.shape):

            print(
                "Warning:",
                self.type,
                "at \n",
                self.data[-1].subject,
                self.data[-1].startindex,
                "shaped inconsistently with dataset.")
            print("Removing it from dataset.")
            self.data.pop(-1)

    def Balance(
        self,
        parentPath,
        filter_band="nofilter",
        ref_folders=config.ref_folders,
            all=False):
        """
        Knowing that reference groups are named as follows:
            - ref 24-30
            - ref 31-40
            - ref 81+
            - ...

        Balances the classes of a dataset such that Classifier.data
        contains an equal number of control and condition-positive
        Spectra or Contig objects. New data are added with Classifier.LoadData

        Parameters:
            - parentPath (required positional): parent path of reference
              folders listed above
        """
        dataFolder = self.trial_name

        totalpos = 0

        totalneg = 0

        for item in self.data:

            if item.group == 1:

                totalneg += 1

            elif item.group > 1:

                totalpos += 1

        if totalpos > totalneg:
            fill = (totalpos - totalneg)
        elif totalneg > totalpos:
            fill = (totalneg - totalpos)

        filled_subs = []

        i = 0
        j = 0
        while i < fill:

            folder = ref_folders[j]

            fnames = os.listdir(parentPath+"/"+folder+"/"+dataFolder)

            subjects = list(set(
                [fname[:config.participantNumLen] for fname in fnames if
                    (fname[:config.participantNumLen]
                    not in config.exclude_subs)]))

            random.shuffle(subjects)

            h = 0
            while (subjects[h], folder) in filled_subs:
                h += 1

            sub_fnames = [
                fname for fname in fnames if
                subjects[h] == fname[:config.participantNumLen]]

            for sub_fname in sub_fnames:
                reqs = [filter_band, ".evt", ".art"]
                if any(ext in sub_fname for ext in reqs):
                    self.LoadData(
                        parentPath
                        + "/"
                        + folder
                        + "/"
                        + dataFolder
                        + "/"
                        + sub_fname)

                    i += 1

            filled_subs.append((subjects[0], folder))

            if j != len(ref_folders) - 1:
                j += 1

            else:
                j = 0

    def LDA(
            self,
            normalize='standard',
            tt_split=0.33,
            lowbound=0,
            highbound=25,
            plot_data=False):
        """
        Linear Discriminant Analysis

        Parameters:
            - normalize: 'standard', 'minmax', or None
            - tt_split: (float) default 0.33
            - lowbound: (int) default 3
            - highbound: (int) default 20
            - plot_data: (bool) default False
        """

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # shuffle dataset
        random.shuffle(self.data)

        # total num for each class
        totalpos = 0

        totalneg = 0

        for item in self.data:

            if item.group == 1:

                totalneg += 1

            elif item.group > 1:

                totalpos += 1

        print("Number of negative outcomes:", totalneg)

        print("Number of positive outcomes:", totalpos)

        # make list of subjects in this dataset
        self.subjects = list(set(
            [(item.source, item.subject) for item in self.data]))

        # shuffle subject list randomly and then
        # split data into train and test spectra (no overlapping subjects)
        # "subject-stratified" train-test split
        random.shuffle(self.subjects)

        split = math.floor(len(self.subjects)*tt_split)

        train_subjects = self.subjects[:split*-1]

        test_subjects = self.subjects[split*-1:]

        if normalize == 'standard':

            myscaler = StandardScaler()

        elif normalize == 'minmax':

            myscaler = MinMaxScaler()

        else:

            normalize = None

        if self.type == "spectra":

            # load dataset into np arrays / train & test
            # make label maps for each set:
            # 1 = positive condition,
            # -1 = negative condition
            train_dataset = np.stack(
                    [SpecObj.data[
                        int(lowbound // SpecObj.freq_res):
                        int(highbound // SpecObj.freq_res) + 1]
                        for SpecObj in self.data
                        if (SpecObj.source, SpecObj.subject)
                        in train_subjects])

            train_labels = np.array(
                    [int(1)
                        if (SpecObj.group == 2)
                        else int(-1)
                        for SpecObj in self.data
                        if (SpecObj.source, SpecObj.subject)
                        in train_subjects])

            test_dataset = np.stack(
                    [SpecObj.data[
                        int(lowbound // SpecObj.freq_res):
                        int(highbound // SpecObj.freq_res) + 1]
                        for SpecObj in self.data
                        if (SpecObj.source, SpecObj.subject)
                        in test_subjects])

            test_labels = np.array(
                    [int(1)
                        if (SpecObj.group == 2)
                        else int(-1)
                        for SpecObj in self.data
                        if (SpecObj.source, SpecObj.subject)
                        in test_subjects])

        # reshape datasets
        nsamples, nx, ny = train_dataset.shape

        train_dataset = train_dataset.reshape((nsamples, nx*ny))

        nsamples, nx, ny = test_dataset.shape

        test_dataset = test_dataset.reshape((nsamples, nx*ny))

        # << Normalize features (z-score standardization) >> #
        # create classifier pipeline
        if normalize is not None:

            clf = make_pipeline(
                    myscaler,
                    LinearDiscriminantAnalysis())

        else:

            clf = make_pipeline(
                    LinearDiscriminantAnalysis())

        # fit to train data

        clf.fit(train_dataset, train_labels)

        print('Classification accuracy on validation data: {:.3f}'.format(
            clf.score(test_dataset, test_labels)))

        if plot_data is True:
            # Colormap
            cmap = colors.LinearSegmentedColormap(
                'red_blue_classes',
                {'red': [(0, 1, 1), (1, 0.7, 0.7)],
                 'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
                 'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
            plt.cm.register_cmap(cmap=cmap)

            # Generate datasets
            def dataset_fixed_cov():
                '''
                Generate 2 Gaussians samples
                with the same covariance matrix
                '''
                n, dim = 300, 2
                np.random.seed(0)
                C = np.array([[0., -0.23], [0.83, .23]])
                X = np.r_[np.dot(np.random.randn(n, dim), C),
                          np.dot(np.random.randn(n, dim), C)
                          + np.array([1, 1])]
                y = np.hstack((np.zeros(n), np.ones(n)))
                return X, y

            def dataset_cov():
                '''
                Generate 2 Gaussians samples
                with different covariance matrices
                '''
                n, dim = 300, 2
                np.random.seed(0)
                C = np.array([[0., -1.], [2.5, .7]]) * 2.
                X = np.r_[np.dot(np.random.randn(n, dim), C),
                          np.dot(np.random.randn(n, dim), C.T)
                          + np.array([1, 4])]
                y = np.hstack((np.zeros(n), np.ones(n)))
                return X, y

            # Plot functions
            def plot_data(lda, X, y, y_pred, fig_index):
                splot = plt.subplot(2, 2, fig_index)
                if fig_index == 1:
                    plt.title('Linear Discriminant Analysis')
                    plt.ylabel('Data with\n fixed covariance')
                elif fig_index == 2:
                    plt.title('Quadratic Discriminant Analysis')
                elif fig_index == 3:
                    plt.ylabel('Data with\n varying covariances')

                tp = (y == y_pred)  # True Positive
                tp0, tp1 = tp[y == 0], tp[y == 1]
                X0, X1 = X[y == 0], X[y == 1]
                X0_tp, X0_fp = X0[tp0], X0[~tp0]
                X1_tp, X1_fp = X1[tp1], X1[~tp1]

                # class 0: dots
                plt.scatter(
                    X0_tp[:, 0],
                    X0_tp[:, 1],
                    marker='.',
                    color='red')

                plt.scatter(
                    X0_fp[:, 0],
                    X0_fp[:, 1],
                    marker='x',
                    s=20,
                    color='#990000')  # dark red

                # class 1: dots
                plt.scatter(
                    X1_tp[:, 0],
                    X1_tp[:, 1],
                    marker='.',
                    color='blue')

                plt.scatter(
                    X1_fp[:, 0],
                    X1_fp[:, 1],
                    marker='x',
                    s=20,
                    color='#000099')  # dark blue

                # class 0 and 1 : areas
                nx, ny = 200, 100
                x_min, x_max = plt.xlim()
                y_min, y_max = plt.ylim()
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, nx),
                    np.linspace(y_min, y_max, ny))

                Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
                Z = Z[:, 1].reshape(xx.shape)
                plt.pcolormesh(
                    xx,
                    yy,
                    Z,
                    cmap='red_blue_classes',
                    norm=colors.Normalize(0., 1.),
                    zorder=0)

                plt.contour(
                    xx,
                    yy,
                    Z,
                    [0.5],
                    linewidths=2.,
                    colors='white')

                # means
                plt.plot(
                    lda.means_[0][0],
                    lda.means_[0][1],
                    '*',
                    color='yellow',
                    markersize=15,
                    markeredgecolor='grey')

                plt.plot(
                    lda.means_[1][0],
                    lda.means_[1][1],
                    '*',
                    color='yellow',
                    markersize=15,
                    markeredgecolor='grey')

                return splot

            def plot_ellipse(splot, mean, cov, color):
                v, w = linalg.eigh(cov)
                u = w[0] / linalg.norm(w[0])
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                # filled Gaussian at 2 standard deviation
                ell = mpl.patches.Ellipse(
                    mean,
                    2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                    180 + angle,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2)

                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.2)
                splot.add_artist(ell)
                splot.set_xticks(())
                splot.set_yticks(())

            def plot_lda_cov(lda, splot):
                plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
                plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')

            def plot_qda_cov(qda, splot):
                plot_ellipse(splot, qda.means_[0], qda.covariance_[0], 'red')
                plot_ellipse(splot, qda.means_[1], qda.covariance_[1], 'blue')

            plt.figure(figsize=(10, 8), facecolor='white')
            plt.suptitle(
                'Linear Discriminant Analysis vs \
                Quadratic Discriminant Analysis',
                y=0.98,
                fontsize=15)

            for i, (X, y) in enumerate([
                                        dataset_fixed_cov(),
                                        dataset_cov()]):

                # Linear Discriminant Analysis
                lda = LinearDiscriminantAnalysis(
                    solver="svd",
                    store_covariance=True)

                y_pred = lda.fit(X, y).predict(X)
                splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
                plot_lda_cov(lda, splot)
                plt.axis('tight')

                # Quadratic Discriminant Analysis
                qda = QuadraticDiscriminantAnalysis(store_covariance=True)
                y_pred = qda.fit(X, y).predict(X)
                splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
                plot_qda_cov(qda, splot)
                plt.axis('tight')
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.show()

        return clf, clf.predict(test_dataset), test_labels

    def SVM(
            self,
            C=1,
            kernel='linear',
            normalize=None,
            iterations=1000,
            plot_PR=False,
            plot_Features=False,
            feat_select=False,
            num_feats=10,
            lowbound=0,
            highbound=25,
            tt_split=0.33):

        """
        Support Vector Machine classifier using scikit-learn base

        Parameters:
            - C: (float) default 1
            - kernel: 'linear' or 'rbf'
            - normalize: 'standard', 'minmax', or None
            - iterations: (int) default 1000
            - plot_PR: (bool) default False
            - plot_Features: (bool) default False
            - feat_select: (bool) default False
            - num_feats: (int) default 10
            - lowbound: (int) default 3
            - highbound: (int) default 20
            - tt_split: (float) default 0.33
        """

        from sklearn.svm import LinearSVC
        from sklearn.svm import SVC
        from sklearn import svm, metrics

        # shuffle dataset
        random.shuffle(self.data)

        Features = np.array(self.data[0].freq[
            int(lowbound // self.data[0].freq_res):
            int(highbound // self.data[0].freq_res) + 1])

        # total num for each class

        totalpos = 0

        totalneg = 0

        for item in self.data:

            if item.group == 1:

                totalneg += 1

            elif item.group == 2:

                totalpos += 1

        print("Number of negative outcomes:", totalneg)

        print("Number of positive outcomes:", totalpos)

        # make list of subjects in this dataset
        self.subjects = list(set([
            (item.source, item.subject)
            for item in self.data]))

        # shuffle subject list randomly and then
        # split data into train and test spectra (no overlapping subjects)
        # "subject-stratified" train-test split
        random.shuffle(self.subjects)

        split = math.floor(len(self.subjects)*tt_split)

        train_subjects = self.subjects[:split*-1]

        test_subjects = self.subjects[split*-1:]

        if normalize == 'standard':

            myscaler = StandardScaler()

        elif normalize == 'minmax':

            myscaler = MinMaxScaler()

        else:

            normalize = None

        if self.type == "spectra":
            # load dataset into np arrays / train & test
            # make label maps for each set:
            # 1 = positive condition,
            # -1 = negative condition
            train_dataset = np.stack(
                [SpecObj.data[
                    int(lowbound // SpecObj.freq_res):
                    int(highbound // SpecObj.freq_res) + 1]
                    for SpecObj in self.data
                    if (SpecObj.source, SpecObj.subject) in train_subjects])

            train_labels = np.array(
                [int(1) if (SpecObj.group == 2) else int(-1)
                    for SpecObj in self.data
                    if (SpecObj.source, SpecObj.subject) in train_subjects])

            test_dataset = np.stack(
                [SpecObj.data[
                    int(lowbound // SpecObj.freq_res):
                    int(highbound // SpecObj.freq_res) + 1]
                    for SpecObj in self.data
                    if (SpecObj.source, SpecObj.subject) in test_subjects])

            test_labels = np.array(
                [int(1) if (SpecObj.group == 2) else int(-1)
                    for SpecObj in self.data
                    if (SpecObj.source, SpecObj.subject) in test_subjects])

        print("Number of samples in train:", train_dataset.shape[0])

        print("Number of samples in test:", test_dataset.shape[0])

        # reshape datasets
        nsamples, nx, ny = train_dataset.shape

        train_dataset = train_dataset.reshape((nsamples, nx*ny))

        nsamples, nx, ny = test_dataset.shape

        test_dataset = test_dataset.reshape((nsamples, nx*ny))

        # << Normalize features (z-score standardization) >> #
        # create classifier pipeline
        if normalize is not None:

            if kernel == 'linear':

                clf = make_pipeline(
                        myscaler,
                        LinearSVC(
                            C=C,
                            random_state=0,
                            max_iter=iterations,
                            dual=False))

            elif kernel == 'rbf':

                clf = make_pipeline(
                        myscaler,
                        SVC(
                            kernel='rbf',
                            C=C, random_state=0,
                            max_iter=iterations))

        else:

            if kernel == 'linear':

                clf = make_pipeline(
                        LinearSVC(
                            C=C,
                            random_state=0,
                            max_iter=iterations,
                            dual=False))

            elif kernel == 'rbf':

                clf = make_pipeline(
                        SVC(
                            kernel='rbf',
                            C=C, random_state=0,
                            max_iter=iterations))

        # fit to train data
        clf.fit(train_dataset, train_labels)

        print('Classification accuracy on validation data: {:.3f}'.format(
            clf.score(test_dataset, test_labels)))

        # Compare to the weights of an SVM
        # only available using linear kernel
        if kernel == "linear":

            svm_weights = np.abs(clf[-1].coef_).sum(axis=0)

            svm_weights /= svm_weights.sum()

        if plot_PR is True:
            # test_score = clf.decision_function(test_dataset)
            # average_precision =
            # average_precision_score(test_dataset, test_score)

            # print('Average precision-recall score:\
            # {0:0.2f}'.format(average_precision))

            disp = plot_precision_recall_curve(clf, test_dataset, test_labels)

            disp.ax_.set_title('2-class Precision-Recall curve: '
                               'AP={0:0.2f}'.format(average_precision))

        # univariate feature selection with F-test for feature scoring
        if feat_select is True:

            # We use the default selection function to select the ten
            # most significant features
            selector = SelectKBest(f_classif, k=num_feats)

            selector.fit(train_dataset, train_labels)

            scores = -np.log10(selector.pvalues_)

            scores /= scores.max()

            # Compare to the selected-weights of an SVM
            if normalize is not None:

                if kernel == 'linear':

                    clf_selected = make_pipeline(
                            SelectKBest(f_classif, k=num_feats),
                            myscaler,
                            LinearSVC(
                                C=C,
                                random_state=0,
                                max_iter=iterations,
                                dual=False))

                elif kernel == 'rbf':
                    clf_selected = make_pipeline(
                            SelectKBest(f_classic, k=num_feats),
                            myscaler,
                            SVC(kernel='rbf',
                                C=C,
                                random_state=0,
                                max_iter=iterations))

            else:

                if kernel == 'linear':

                    clf_selected = make_pipeline(
                            SelectKBest(f_classif, k=num_feats),
                            LinearSVC(
                                C=C,
                                random_state=0,
                                max_iter=iterations,
                                dual=False))

                elif kernel == 'rbf':
                    clf_selected = make_pipeline(
                            SelectKbest(f_classic, k=num_feats),
                            SVC(
                                kernel='rbf',
                                C=C,
                                random_state=0,
                                max_iter=iterations))

            clf_selected.fit(train_dataset, train_labels)

            print('Classification accuracy after univariate feature selection:\
                    {:.3f}'.format(clf_selected.score(
                        test_dataset,
                        test_labels)))

            # Compare to the weights of an SVM
            # only available using linear kernel
            if kernel == 'linear':

                svm_weights_selected = np.abs(
                    clf_selected[-1].coef_).sum(axis=0)

                svm_weights_selected /= svm_weights_selected.sum()

        if plot_Features is True:

            import Plots

            if kernel == 'linear':

                if feat_select is False:
                    Plots.plot_svm_features(
                        Features,
                        svm_weights,
                        network_channels=self.network_channels)

                elif feat_select is True:
                    Plots.plot_svm_features(
                        Features,
                        svm_weights_selected,
                        scores=scores,
                        network_channels=self.network_channels)

        return clf, clf.predict(test_dataset), test_labels

    def CNN(
        self,
        normalize=None,
        learning_rate=0.01,
        lr_decay=False,
        beta1=0.9,
        beta2=0.999,
        epochs=100,
        plot_ROC=False,
        tt_split=0.33,
            k_fold=None):

        """
        Convolutional Neural Network classifier, using Tensorflow base

        Parameters:
            - normalize: 'standard', 'minmax', or None
                method with which contig data will be normalized
                note: within itself, not relative to baseline or other
            - learning_rate: (float) 0.01
                how quickly the weights adapt at each iteration
            - lr_decay: (bool) default True
                whether the learning rate should decay at a rate of 0.96
            - beta1: (float) default 0.9
                beta1 value for Adam optimizer
            - beta2: (float) default 0.999
                beta2 value for Adam optimizer
            - epochs: (int) default 100
                number of training iterations
            - plot_ROC: (bool) default False
                evaluates model on validation data and plots ROC curve
            - tt_split: (float) default 0.33
                ratio of subjects (from each group, control or positive)
                which is to be reserved from training for validation data
            - k_fold: (tuple ints) default None
                cross-validation fold i and total folds k
                ex: (2, 5) fold 2 of 5
        """
        # quiet some TF warnings
        TF_CPP_MIN_LOG_LEVEL = 2
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras import regularizers
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.layers import Conv2DTranspose
        from tensorflow.keras.layers import LeakyReLU
        from tensorflow.keras.layers import UpSampling2D
        from tensorflow.keras.layers import MaxPooling2D
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import BatchNormalization
        import datetime

        # shuffle dataset
        random.shuffle(self.data)

        # total num for each class
        totalpos = 0
        totalneg = 0
        for item in self.data:
            if item.group == 1:
                totalneg += 1
            elif item.group > 1:
                totalpos += 1

        if k_fold is None:
            print("Number of negative outcomes:", totalneg)
            print("Number of positive outcomes:", totalpos)

        # make list of subjects in this dataset
        self.subjects = list(set([
            (item.source, item.subject)
            for item in self.data]))

        if k_fold is None:
            print("Total number of subjects:", len(self.subjects))

        # get num condition positive
        j = 0
        for sub in self.subjects:
            if int(sub[1][0]) > 1:
                j += 1

        if k_fold is None:
            print(
                "% Positive in all subjects:",
                j / len(self.subjects))
            print(
                "% Negative in all subjects:",
                (len(self.subjects) - j) / len(self.subjects))

        train_subjects = []
        test_subjects = []

        # shuffle subject list randomly and then
        # split data into train and test spectra (no overlapping subjects)
        # "subject-stratified" train-test split
        if k_fold is None:
            random.shuffle(self.subjects)

            split = math.floor(len(self.subjects)*tt_split)

            pos_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) > 1]
            pos_split = math.floor(len(pos_subjects)*tt_split)

            neg_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) == 1]
            neg_split = math.floor(len(neg_subjects)*tt_split)

            train_subjects = pos_subjects[:pos_split*-1]
            for sub in neg_subjects[:neg_split*-1]:
                train_subjects.append(sub)

            test_subjects = pos_subjects[pos_split*-1:]
            for sub in neg_subjects[neg_split*-1:]:
                test_subjects.append(sub)

        else:
            from sklearn.model_selection import KFold

            self.subjects.sort()

            # the first n_samples % n_splits folds have size:
            # n_samples // n_splits + 1, other folds have size:
            # n_samples // n_splits
            kf = KFold(n_splits=k_fold[1])

            # first look at only subjects with subject code above 1 (cond-pos)
            pos_subjects = [sub for sub in self.subjects if int(sub[1][0]) > 1]
            train_indexes, test_indexes = list(
                kf.split(pos_subjects))[k_fold[0]]

            for i, sub in enumerate(pos_subjects):
                if i in train_indexes:
                    train_subjects.append(sub)
                elif i in test_indexes:
                    test_subjects.append(sub)
                else:
                    print("There was an error in the k-fold algorithm.")
                    print("Exiting.")
                    sys.exit(1)

            # then look at subjects with subject code "1" (condition-neg)
            neg_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) == 1]
            train_indexes, test_indexes = list(
                kf.split(neg_subjects))[k_fold[0]]

            for i, sub in enumerate(neg_subjects):
                if i in train_indexes:
                    train_subjects.append(sub)
                elif i in test_indexes:
                    test_subjects.append(sub)
                else:
                    print("There was an error in the k-fold algorithm.")
                    print("Exiting.")
                    sys.exit(1)

        train_dataset = np.expand_dims(np.stack(
            [ContigObj.data
                for ContigObj in self.data
                if (ContigObj.source, ContigObj.subject) in train_subjects]),
            -1)

        train_labels = np.array(
            [int(1) if (ContigObj.group > 1) else int(0)
                for ContigObj in self.data
                if (ContigObj.source, ContigObj.subject) in train_subjects])

        test_dataset = np.expand_dims(np.stack(
            [ContigObj.data
                for ContigObj in self.data
                if (ContigObj.source, ContigObj.subject) in test_subjects]),
            -1)

        test_labels = np.array(
            [int(1) if (ContigObj.group > 1) else int(0)
                for ContigObj in self.data
                if (ContigObj.source, ContigObj.subject) in test_subjects])

        if k_fold is None:
            print("Number of samples in train:", train_dataset.shape[0])
            print("Number of samples in test:", test_dataset.shape[0])

        num_pos_train_samps = 0
        for label in train_labels:
            if label == 1:
                num_pos_train_samps += 1

        num_pos_test_samps = 0
        for label in test_labels:
            if label == 1:
                num_pos_test_samps += 1

        if k_fold is None:
            print(
                "% Positive samples in train:",
                num_pos_train_samps / len(train_labels))
            print(
                "% Positive samples in test:",
                num_pos_test_samps / len(test_labels))

        # introduce equential set
        model = tf.keras.models.Sequential()

        # 1
        model.add(BatchNormalization())

        model.add(Conv2D(
            5,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu',
            data_format='channels_last'))

        # model.add(LeakyReLU(alpha=0.1))

        model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=1,
            padding='valid',
            data_format='channels_last'))

        # 2
        model.add(BatchNormalization())

        model.add(Conv2D(
            5,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu',
            data_format='channels_last'))

        # model.add(LeakyReLU(alpha=0.1))

        model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=1,
            padding='valid',
            data_format='channels_last'))

        # 3
        model.add(BatchNormalization())

        model.add(Conv2D(
            5,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu',
            data_format='channels_last'))

        # model.add(LeakyReLU(alpha=0.1))

        model.add(MaxPooling2D(
            pool_size=(5, 5),
            strides=1,
            padding='valid',
            data_format='channels_last'))

        # 4
        model.add(BatchNormalization())

        model.add(Conv2D(
            5,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu',
            data_format='channels_last'))

        # model.add(LeakyReLU(alpha=0.1))

        model.add(MaxPooling2D(
            pool_size=(5, 5),
            strides=1,
            padding='valid',
            data_format='channels_last'))

        # 5
        model.add(BatchNormalization())

        model.add(Conv2D(
            5,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu',
            data_format='channels_last'))

        # model.add(LeakyReLU(alpha=0.1))

        model.add(MaxPooling2D(
            pool_size=(5, 5),
            strides=1,
            padding='valid',
            data_format='channels_last'))

        # dropout
        # model.add(Dropout(0.2))

        # flatten
        model.add(Flatten(data_format='channels_last'))

        model.add(Dense(10, activation='softmax'))
        model.add(Dense(2, activation='softmax'))

        # build model
        model.build(train_dataset.shape)

        # print model summary at buildtime
        if k_fold is None:
            model.summary()

            print("Input shape:", train_dataset.shape)

        # adaptive learning rate
        if lr_decay is True:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=100,
                decay_rate=0.96)

        # model compilation
        model.compile(
            # optimizer=tf.keras.optimizers.Adam(
            #     learning_rate=learning_rate),
            optimizer=tf.keras.optimizers.Adagrad(
                learning_rate=learning_rate,
                initial_accumulator_value=0.1,
                epsilon=1e-07,
                name='Adagrad'),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        # tensorboard setup
        log_dir = 'logs/fit/'\
            + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1)

        history = model.fit(
            train_dataset,
            train_labels,
            epochs=epochs,
            validation_data=(test_dataset, test_labels),
            batch_size=32,
            callbacks=[tensorboard_callback],
            # verbose=0 if k_fold is not None else 1
        )

        y_pred_keras = model.predict(test_dataset)[:, 0]

        if plot_ROC is True:
            from Plots import roc
            roc(y_pred_keras, test_labels, fname=log_dir+"/ROC")

        return(model, y_pred_keras, test_labels)

    def KfoldCrossVal(
        self,
        ML_function,
            k=1):
        """
        Resampling procedure used to evaluate ML models
        on a limited data sample

        Parameters:
            - k (default 5): int
                number of groups that a given data sample will be split into
        """
        all_y_preds = []
        all_y_labels = []
        all_aucs = []

        from Plots import roc

        f = open(self.type+"_"+os.path.basename(self.trial_name)+".txt", 'w')

        for i in tqdm(range(k)):
            model, y_pred, y_labels = self.ML_function(k_fold=(i, k))
            # if model_type == 'CNN':
            #     hist, model, y_pred, y_labels = self.CNN(
            #         k_fold=(i, k),
            #         plot_ROC=True)
            #
            # elif model_type == 'SVM':
            #     model, y_pred, y_labels = self.SVM(
            #         kernel='linear',
            #         iterations=1000,
            #         normalize=None,
            #         num_feats=30,
            #         feat_select=False,
            #         plot_Features=True,
            #         lowbound=0,
            #         highbound=25)
            #
            # elif model_type == 'LDA':
            #     model, y_pred, y_labels = self.LDA(
            #         normalize=None,
            #         lowbound=0,
            #         highbound=25)

            for pred, label in zip(y_pred, y_labels):
                all_y_preds.append(pred)
                all_y_labels.append(label)

            auc = roc(y_pred, y_labels, plot=False)
            if auc < 0.5:
                auc = 1 - auc

            all_aucs.append(auc)
            f.write(str(auc))
            f.write('\n')

        f.close()

        auc = roc(
            all_y_preds,
            all_y_labels,
            fname=self.type + "_" + self.trial_name)

        print(all_aucs)

        return roc
