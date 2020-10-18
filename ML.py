import os
import numpy as np
import config
# import scipy
import random
import math
from tqdm import tqdm
import sklearn
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from Prep import Contig, Spectra


# Utility function to move the midpoint of a colormap to be around
# the values of interest
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):

        self.midpoint = midpoint

        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))


# takes one positional argument: either "contigs" or "spectra"
class Classifier:

    def __init__(self, type, network_channels=config.network_channels):

        self.data = []

        self.type = type

        self.network_channels = config.network_channels

        self.trial_name = None

    def LoadData(self, path):
        fname = os.path.basename(path)

        self.trial_name = os.path.basename(
            os.path.dirname(
                os.path.dirname(path)))\
            + "/"+os.path.basename(os.path.dirname(path))

        if self.type == "contigs":
            self.data.append(
                Contig(
                    np.genfromtxt(path, delimiter=","),
                    fname.split('_')[2][:-4],
                    fname[:config.participantNumLen],
                    fname.split('_')[1]))

        elif self.type == "spectra":
            self.data.append(
                Spectra(
                    np.genfromtxt(path, delimiter=",").T[1:].T,
                    np.genfromtxt(path, delimiter=",").T[0].T,
                    fname.split('_')[2][:-4], fname[:config.participantNumLen],
                    fname.split('_')[1],
                    source=os.path.basename(os.path.dirname(path))))

    def Balance(self, parentPath):
        folders = [
                    "ref 24-30",
                    "ref 31-40",
                    "ref 41-50",
                    "ref 51-60",
                    "ref 61-70",
                    "ref 71-80",
                    "ref 81+"]

        spectralFolder = self.trial_name

        totalpos = 0

        totalneg = 0

        for item in self.data:

            if item.group == 1:

                totalneg += 1

            elif item.group == 2:

                totalpos += 1

        fill = (totalpos - totalneg) / len(folders)

        for folder in tqdm(folders):

            fnames = os.listdir(parentPath+"/"+folder+"/"+spectralFolder)

            random.shuffle(fnames)

            i = 0

            while i < fill:

                self.LoadData(
                    parentPath
                    + "/"
                    + folder
                    + "/"
                    + spectralFolder
                    + "/"
                    + fnames[i])

                i += 1

    def LDA(
            self,
            normalize='standard',
            tt_split=0.33,
            lowbound=3,
            highbound=20,
            plot_data=False):

        # shuffle dataset
        random.shuffle(self.data)

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
        subjects = list(set(
            [(item.source, item.subject) for item in self.data]))

        # shuffle subject list randomly and then
        # split data into train and test spectra (no overlapping subjects)
        # "subject-stratified" train-test split
        random.shuffle(subjects)

        split = math.floor(len(subjects)*tt_split)

        train_subjects = subjects[:split*-1]

        test_subjects = subjects[split*-1:]

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
                        int(lowbound // SpecObj.freq_res) + 1:
                        int(highbound // SpecObj.freq_res) + 2]
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
                        int(lowbound // SpecObj.freq_res) + 1:
                        int(highbound // SpecObj.freq_res) + 2]
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
            lowbound=3,
            highbound=20,
            tt_split=0.33):

        # shuffle dataset
        random.shuffle(self.data)

        Features = np.array(self.data[0].freq[
            int(lowbound // self.data[0].freq_res) + 1:
            int(highbound // self.data[0].freq_res) + 2])

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
        subjects = list(set([
            (item.source, item.subject)
            for item in self.data]))

        # shuffle subject list randomly and then
        # split data into train and test spectra (no overlapping subjects)
        # "subject-stratified" train-test split
        random.shuffle(subjects)

        split = math.floor(len(subjects)*tt_split)

        train_subjects = subjects[:split*-1]

        test_subjects = subjects[split*-1:]

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
                    int(lowbound // SpecObj.freq_res) + 1:
                    int(highbound // SpecObj.freq_res) + 2]
                    for SpecObj in self.data
                    if (SpecObj.source, SpecObj.subject) in train_subjects])

            train_labels = np.array(
                [int(1) if (SpecObj.group == 2) else int(-1)
                    for SpecObj in self.data
                    if (SpecObj.source, SpecObj.subject) in train_subjects])

            test_dataset = np.stack(
                [SpecObj.data[
                    int(lowbound // SpecObj.freq_res) + 1:
                    int(highbound // SpecObj.freq_res) + 2]
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

            if kernel == 'linear':
                # set up figure and axes (rows)
                fig, axs = plt.subplots(
                    nrows=len(self.network_channels),
                    figsize=(20, 40))

                plt.rcParams['figure.dpi'] = 200

                # plt.clf()

                # X_indices = np.arange(train_dataset.shape[-1])
                X_indices = Features

                i = 0

                j = 0

                for channel in self.network_channels:

                    axs[j].set_title(channel)

                    axs[j].bar(
                            X_indices - .25,
                            svm_weights[i:i + len(X_indices)],
                            width=.1, label='SVM weight')

                    if feat_select is not True:

                        axs[j].legend()

                    i += len(X_indices)

                    j += 1

                if feat_select is True:

                    i = 0

                    j = 0

                    for channel in self.network_channels:

                        axs[j].bar(
                            X_indices - .45,
                            scores[i:i+len(X_indices)],
                            width=.1,
                            label=r'Univariate score ($-Log(p_{value})$)')

                        axs[j].legend()

                        i += len(X_indices)

                        j += 1

                axs[0].set_title("Comparing feature selection")

                axs[-1].set_xlabel('Feature number (in Hz)')

                # plt.yticks(())
                fig.tight_layout()

                plt.show()

        def CNN(
            self,
            normalize=None,
            learning_rate=0.01,
            lr_decay=True,
            beta1=0.9,
            beta2=0.99,
            epochs=100,
            plot_ROC=False,
            tt_split=0.33):

            import tensorflow as tf
            from tensorflow.keras import Model
            from tensorflow.keras import regularizers
            from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, UpSampling2d, MaxPooling2D, Dropout, BatchNormalization
            import datetime

            # shuffle dataset
            random.shuffle(self.data)

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
            subjects = list(set([
                (item.source, item.subject)
                for item in self.data]))

            # shuffle subject list randomly and then
            # split data into train and test spectra (no overlapping subjects)
            # "subject-stratified" train-test split
            random.shuffle(subjects)

            split = math.floor(len(subjects)*tt_split)

            train_subjects = subjects[:split*-1]

            test_subjects = subjects[split*-1:]

            train_dataset = np.stack(
                [ContigObj.data
                    for ContigObj in self.data
                    if (ContigObj.source, ContigObj.subject) in train_subjects])

            train_labels = np.array(
                [int(1) if (ContigObj.group == 2) else int(-1)
                    for ContigObj in self.data
                    if (ContigObj.source, ContigObj.subject) in train_subjects])

            test_dataset = np.stack(
                [ContigObj.data
                    for ContigObj in self.data
                    if (ContigObj.source, ContigObj.subject) in test_subjects])

            test_labels = np.array(
                [int(1) if (ContigObj.group == 2) else int(-1)
                    for ContigObj in self.data
                    if (ContigObj.source, ContigObj.subject) in test_subjects])

            print("Number of samples in train:", train_dataset.shape[0])

            print("Number of samples in test:", test_dataset.shape[0])

            # introduce equential set
            model = tf.keras.models.Sequential()

            # temporal convolution
            model.add(Conv2D(10, kernel_size=(50, 1), strides=1, padding='same', activation='relu', data_format='channels_last', kernel_initializer='glorot_uniform'))

            # spatial filter
            model.add(Conv2D(5, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last', kernel_initializer='glorot_uniform'))

            # pooling and dropout
            model.add(MaxPooling2D(pool_size=(3, 3), strides=3, padding='same', data_format='channels_last'))

            model.add(Dropout(0.5))

            # flatten
            model.add(Flatten(data_format='channels_last'))

            # batch normalize
            model.add(BatchNormalization())

            # dense layers
            model.add(Dense(25, activation='sigmoid', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

            model.add(Dense(2, activation='sigmoid', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

            # build model
            model.build(self.data[0].data.shape)

            # print model summary at buildtime
            model.summary()
            print("Input shape:", self.data[0].data.shape)

            # adaptive learning rate
            if lr_decay==True:
                learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=learning_rate,
                    decay_steps=100,
                    decay_rate=0.96)

            # model compilation
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    beta_1=beta1,
                    beta_2=beta2),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

            # tensorboard setup
            log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            history = model.fit(
                train_dataset,
                train_labels,
                epochs=epochs,
                validation_data=(test_dataset, test_labels),
                batch_size=32,
                callbacks=[tensorboard_callback]
            )

            return(history, model)
