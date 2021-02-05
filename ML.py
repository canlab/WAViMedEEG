import os
import numpy as np
import config
import random
import math
from tqdm import tqdm
import sys

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

        self.groups = []

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

        if self.data[-1].group not in self.groups:
            self.groups.append(self.data[-1].group)
        self.groups.sort()

    def Balance(
        self,
            verbosity=True):
        """
        Balances the classes of a dataset such that Classifier.data
        contains an equal number of control and condition-positive
        Spectra or Contig objects.
        """
        # pop off data from the class with less data until class sizes are equal
        groups = []
        for dataObj in self.data:
            if dataObj.group not in groups:
                groups.append(dataObj.group)

        group_sizes = []
        for group in groups:
            group_size = 0
            for dataObj in self.data:
                if dataObj.group == group:
                    group_size+=1
            group_sizes.append(group_size)

        random.shuffle(self.data)

        if len(groups) != 2:
            raise ValueError
        else:
            larger_group = group_sizes.index(np.max(group_sizes))
            smaller_group = group_sizes.index(np.min(group_sizes))

        if verbosity:
            print("Before balancing:")
            for i, group in enumerate(groups):
                print(
                    "Type:",
                    config.group_names[group],
                    "\tAmount:",
                    group_sizes[i])

        i = 0
        while group_sizes[larger_group] > group_sizes[smaller_group]:
            if self.data[i].group == groups[larger_group]:
                self.data.pop(i)
                group_sizes[larger_group]-=1
            else:
                i += 1

        if verbosity:
            print("After balancing:")
            for i, group in enumerate(groups):
                print(
                    "Type:",
                    config.group_names[group],
                    "\tAmount:",
                    group_sizes[i])

    def Prepare(
        self,
        k_fold=None,
        verbosity=True,
        tt_split=0.33,
        labels=None,
            normalize=None):
        """
        Prepares data within Classifier object such that all in Classifier.data
        are contained in either self.train_dataset or self.test_dataset,
        with the corresponding subjects listed in self.train_subjects and
        self.test_subjects. This is done either according to a k_fold cross-
        evaluation schema, if k_fold is provided, otherwise is split randomly
        by the float provided in tt_split.

            - tt_split: (float, default 0.33)
                ratio of subjects (from each group, control or positive)
                which is to be reserved from training for validation data
            - k_fold: (tuple ints, default None)
                cross-validation fold i and total folds k
                ex: (2, 5) fold 2 of 5
            - verbosity: (bool, default True) whether to print information
                such as the ratios of condition positive-to-negative subjects,
                total number of samples loaded into each set, etc.
            - labels: (list, default None) the real set of false and
                position condition labels, if only loading in one class
                under the context
            - normalize: 'standard', 'minmax', or None
                method with which contig data will be normalized
                note: within itself, not relative to baseline or other
        """

        # shuffle dataset
        random.shuffle(self.data)

        # total num for each class
        totalpos = 0
        totalneg = 0

        if len(self.groups) > 2:
            print("Warning: Not correct for # groups > 2 yet. Coming soon.")

        if labels is None:
            self.groups.sort()
        else:
            self.groups = list(labels)

        self.group_names = [config.group_names[group] for group in self.groups]

        if any(
            num not in config.group_names\
            for num in self.groups):
            print("Invalid group name encountered, you should have entered it "
            + "in config.group_names and config.group_colors first.")
            raise ValueError
            sys.exit(1)
        if any(
            name not in config.group_names.values()\
            for name in self.group_names):
            print("Invalid group name encountered, you should have entered it "
            + "in config.group_names and config.group_colors first.")
            raise ValueError
            sys.exit(1)

        for item in self.data:
            if item.group == self.groups[0]:
                totalneg += 1
            else:
                totalpos += 1

        if verbosity:
            print("Number of negative outcomes:", totalneg)
            print("Number of positive outcomes:", totalpos)

        # make list of subjects in this dataset
        self.subjects = list(set([
            (item.source, item.subject)
            for item in self.data]))

        if verbosity:
            print("Total number of subjects:", len(self.subjects))

        # get num condition positive
        j = 0
        for sub in self.subjects:
            if int(sub[1][0]) != self.groups[0]:
                j += 1

        if verbosity:
            print(
                "% Positive in all subjects:",
                j / len(self.subjects))
            print(
                "% Negative in all subjects:",
                (len(self.subjects) - j) / len(self.subjects))

        self.train_subjects = []
        self.test_subjects = []

        # shuffle subject list randomly and then
        # split data into train and test spectra (no overlapping subjects)
        # "subject-stratified" train-test split
        if k_fold is None and tt_split > 0:
            random.shuffle(self.subjects)

            split = math.floor(len(self.subjects)*tt_split)

            # TODO
            # not sure if this will work correctly with extremely low
            # val for tt_split, e.g. 0, or 0.001
            pos_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) != self.groups[0]]
            pos_split = math.floor(len(pos_subjects)*tt_split)

            neg_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) == self.groups[0]]
            neg_split = math.floor(len(neg_subjects)*tt_split)

            self.train_subjects = pos_subjects[:pos_split*-1]
            for sub in neg_subjects[:neg_split*-1]:
                self.train_subjects.append(sub)

            self.test_subjects = pos_subjects[pos_split*-1:]
            for sub in neg_subjects[neg_split*-1:]:
                self.test_subjects.append(sub)

        elif k_fold is None and tt_split == 0:
            random.shuffle(self.subjects)

            pos_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) != self.groups[0]]

            neg_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) == self.groups[0]]

            self.train_subjects = pos_subjects
            for sub in neg_subjects:
                self.train_subjects.append(sub)

            self.test_subjects = []

        elif k_fold is None and tt_split == 1:
            random.shuffle(self.subjects)

            pos_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) != self.groups[0]]

            neg_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) == self.groups[0]]

            self.test_subjects = pos_subjects
            for sub in neg_subjects:
                self.test_subjects.append(sub)

            self.train_subjects = []

        else:
            from sklearn.model_selection import KFold

            self.subjects.sort()

            # the first n_samples % n_splits folds have size:
            # n_samples // n_splits + 1, other folds have size:
            # n_samples // n_splits
            kf = KFold(n_splits=k_fold[1])

            # first look at only subjects with subject code above 1 (cond-pos)
            pos_subjects = [sub for sub in self.subjects if int(sub[1][0]) != self.groups[0]]
            train_indexes, test_indexes = list(
                kf.split(pos_subjects))[k_fold[0]]

            for i, sub in enumerate(pos_subjects):
                if i in train_indexes:
                    self.train_subjects.append(sub)
                elif i in test_indexes:
                    self.test_subjects.append(sub)
                else:
                    print("There was an error in the k-fold algorithm.")
                    print("Exiting.")
                    sys.exit(1)

            # then look at subjects with subject code "1" (condition-neg)
            neg_subjects = [
                sub for sub in self.subjects if int(sub[1][0]) == self.groups[0]]
            train_indexes, test_indexes = list(
                kf.split(neg_subjects))[k_fold[0]]

            for i, sub in enumerate(neg_subjects):
                if i in train_indexes:
                    self.train_subjects.append(sub)
                elif i in test_indexes:
                    self.test_subjects.append(sub)
                else:
                    print("There was an error in the k-fold algorithm.")
                    print("Exiting.")
                    sys.exit(1)

        if tt_split != 1:
            self.train_dataset = np.expand_dims(np.stack(
                [ContigObj.data
                    for ContigObj in self.data
                    if (ContigObj.source, ContigObj.subject)
                    in self.train_subjects]),
                -1)

            self.train_labels = np.array(
                [int(1) if (ContigObj.group != self.groups[0]) else int(0)
                    for ContigObj in self.data
                    if (ContigObj.source, ContigObj.subject)
                    in self.train_subjects])

        if tt_split != 0:
            self.test_dataset = np.expand_dims(np.stack(
                [ContigObj.data
                    for ContigObj in self.data
                    if (ContigObj.source, ContigObj.subject)
                    in self.test_subjects]),
                -1)

            self.test_labels = np.array(
                [int(1) if (ContigObj.group != self.groups[0]) else int(0)
                    for ContigObj in self.data
                    if (ContigObj.source, ContigObj.subject)
                    in self.test_subjects])

        else:
            self.test_dataset = np.ndarray(self.train_dataset.shape)
            self.test_labels = np.ndarray(self.train_labels.shape)

        if tt_split == 1:
            self.train_dataset = np.ndarray(self.test_dataset.shape)
            self.train_labels = np.ndarray(self.test_labels.shape)

        if k_fold is None:
            print("Number of samples in train:", self.train_dataset.shape[0])
            print("Number of samples in test:", self.test_dataset.shape[0])

        # normalize / standardize
        if normalize is not None:
            if normalize == 'standard':
                scaler = StandardScaler()

            elif normalize == 'minmax':
                scaler = MinMaxScaler()

            else:
                print("Unsupported type of normalize:", normalize)
                raise ValueError
                sys.exit(3)

            scaler.fit([inputObj.data for inputObj in self.data])
            self.train_dataset = scaler.transform(self.train_dataset)
            self.test_dataset = scaler.transform(self.test_dataset)

        num_pos_train_samps = 0
        for label in self.train_labels:
            if label == 1:
                num_pos_train_samps += 1

        num_pos_test_samps = 0
        for label in self.test_labels:
            if label == 1:
                num_pos_test_samps += 1

        if verbosity:
            print(
                "% Positive samples in train:",
                num_pos_train_samps / len(self.train_labels))
            print(
                "% Positive samples in test:",
                num_pos_test_samps / len(self.test_labels))

    def LDA(
        self,
            plot_data=False):
        """
        Linear Discriminant Analysis

        Parameters:
            - tt_split: (float) default 0.33
            - lowbound: (int) default 3
            - highbound: (int) default 20
            - plot_data: (bool) default False
        """

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # reshape datasets
        nsamples, nx, ny, ndepth = self.train_dataset.shape

        rs_train_dataset = self.train_dataset.reshape((nsamples, nx*ny))

        nsamples, nx, ny, ndepth = self.test_dataset.shape

        rs_test_dataset = self.test_dataset.reshape((nsamples, nx*ny))

        # << Normalize features (z-score standardization) >> #
        # create classifier pipeline
        clf = make_pipeline(
                LinearDiscriminantAnalysis())

        # fit to train data
        clf.fit(rs_train_dataset, self.train_labels)

        print('Classification accuracy on validation data: {:.3f}'.format(
            clf.score(rs_test_dataset, self.test_labels)))

        y_pred = clf.predict(rs_test_dataset)

        if plot_data is True:
            import Plots
            Plots.plot_LDA(
                clf,
                rs_train_dataset,
                self.test_labels, y_pred)

        return clf, y_pred, self.test_labels

    def SVM(
        self,
        C=1,
        kernel='linear',
        iterations=1000,
        plot_PR=False,
        plot_Features=False,
        feat_select=False,
            num_feats=10):
        """
        Support Vector Machine classifier using scikit-learn base

        Parameters:
            - C: (float) default 1
            - kernel: 'linear' or 'rbf'
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

        Features = np.array(self.data[0].freq)

        # reshape datasets
        nsamples, nx, ny, ndepth = self.train_dataset.shape

        rs_train_dataset = self.train_dataset.reshape((nsamples, nx*ny))

        nsamples, nx, ny, ndepth = self.test_dataset.shape

        rs_test_dataset = self.test_dataset.reshape((nsamples, nx*ny))

        # << Normalize features (z-score standardization) >> #
        # create classifier pipeline
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
        clf.fit(rs_train_dataset, self.train_labels)

        print('Classification accuracy on validation data: {:.3f}'.format(
            clf.score(rs_test_dataset, self.test_labels)))

        # Compare to the weights of an SVM
        # only available using linear kernel
        if kernel == "linear":
            svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
            svm_weights /= svm_weights.sum()

        if plot_PR is True:
            disp = plot_precision_recall_curve(
                clf,
                rs_test_dataset,
                self.test_labels)

            disp.ax_.set_title('2-class Precision-Recall curve: '
                               'AP={0:0.2f}'.format(average_precision))

        # univariate feature selection with F-test for feature scoring
        if feat_select is True:

            # We use the default selection function to select the ten
            # most significant features
            selector = SelectKBest(f_classif, k=num_feats)
            selector.fit(rs_train_dataset, self.train_labels)
            scores = -np.log10(selector.pvalues_)
            scores /= scores.max()

            # Compare to the selected-weights of an SVM
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

            clf_selected.fit(rs_train_dataset, self.train_labels)

            print('Classification accuracy after univariate feature selection:\
                    {:.3f}'.format(clf_selected.score(
                        rs_test_dataset,
                        self.test_labels)))

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

        return clf, clf.predict(rs_test_dataset), self.test_labels

    def CNN(
        self,
        learning_rate=0.001,
        lr_decay=False,
        beta1=0.9,
        beta2=0.999,
        epochs=100,
        plot_ROC=False,
        verbosity=True,
        eval=None,
        depth=5,
            regularizer=None):
        """
        Convolutional Neural Network classifier, using Tensorflow base

        Parameters:
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

        # introduce sequential set
        model = tf.keras.models.Sequential()

        for i in range(depth):
            # 1
            model.add(BatchNormalization())

            model.add(Conv2D(
                5,
                kernel_size=(3, 3),
                strides=1,
                padding='same',
                activation='swish',
                data_format='channels_last'))

            model.add(MaxPooling2D(
                pool_size=(3, 3),
                strides=1,
                padding='valid',
                data_format='channels_last'))

        model.add(Dropout(0.5))

        # flatten
        model.add(Flatten(data_format='channels_last'))

        # model.add(Dense(10, activation='softmax'))
        model.add(Dense(
            2,
            activation='softmax',
            kernel_regularizer=regularizer))

        # build model
        model.build(self.train_dataset.shape)

        # print model summary at buildtime
        if verbosity:
            model.summary()
            print("Input shape:", self.train_dataset.shape)

        # adaptive learning rate
        if lr_decay is True:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=100,
                decay_rate=0.96)

        # model compilation
        model.compile(
            optimizer=tf.keras.optimizers.Adagrad(
                learning_rate=learning_rate,
                initial_accumulator_value=0.1,
                epsilon=1e-07,
                name='Adagrad'),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        # tensorboard setup
        checkpoint_dir = 'logs/fit/'\
            + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')\
            + "_"\
            + self.trial_name.replace("/", "_")\

        for group in self.group_names:
            checkpoint_dir = checkpoint_dir + "_" + group

        self.checkpoint_dir = checkpoint_dir

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=checkpoint_dir,
            histogram_freq=1)

        # save model history (acc, loss) to csv file
        csv_logger = tf.keras.callbacks.CSVLogger(
            checkpoint_dir+"/training.log",
            separator=',')

        history = model.fit(
            self.train_dataset,
            self.train_labels,
            epochs=epochs,
            validation_data=(self.test_dataset, self.test_labels) if
            any(val in [1, 2, 3] for val in self.test_labels)
            else None,
            batch_size=32,
            callbacks=[
                tensorboard_callback,
                csv_logger],
            verbose=verbosity)

        f = open(checkpoint_dir+"/summary.txt", 'w')
        f.write(str(model.summary()))
        f.close()

        model.save(checkpoint_dir+"/my_model")

        # save accuracy curve to log dir
        plt.plot(
            history.history['accuracy'],
            label='train: '\
                + "Subs: " + str(len(self.train_subjects)) + " "\
                + "Data: " + str(len(self.train_dataset)))
        if all(val in config.group_names for val in self.test_labels):
            plt.plot(
                history.history['val_accuracy'],
                label='test: '\
                    + "Subs: " + str(len(self.test_subjects)) + " "\
                    + "Data: " + str(len(self.test_dataset)))
        plt.title('Epoch Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.savefig(checkpoint_dir+"/epoch_accuracy")
        plt.clf()

        # save loss curve to log dir
        plt.plot(
            history.history['loss'],
            label='train: '\
                + "Subs: " + str(len(self.train_subjects)) + " "\
                + "Data: " + str(len(self.train_dataset)))
        if all(val in config.group_names for val in self.test_labels):
            plt.plot(
                history.history['val_loss'],
                label='test: '\
                    + "Subs: " + str(len(self.test_subjects)) + " "\
                    + "Data: " + str(len(self.test_dataset)))

        plt.title('Epoch Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.savefig(checkpoint_dir+"/epoch_loss")
        plt.clf()

        y_pred_keras = model.predict(self.test_dataset)[:, 0]

        if plot_ROC is True:
            from Plots import roc
            roc(
                y_pred_keras,
                self.test_labels,
                fname=checkpoint_dir+"/ROC")

        return(model, y_pred_keras, self.test_labels)

    def eval_saved_CNN(
        self,
        checkpoint_dir,
        plot_hist=False,
        fname=None,
            pred_level='all'):

        import tensorflow as tf

        model = tf.keras.models.load_model(checkpoint_dir+"/my_model")

        self.group_names = checkpoint_dir.split('_')[-2:]
        self.groups = []
        for name in self.group_names:
            for key, value in config.group_names.items():
                if name == value:
                    self.groups.append(key)
        self.groups.sort()

        model.summary()

        y_pred_keras = model.predict(self.test_dataset)[:, 1]
        test_set_labels = self.test_labels

        if pred_level == 'subject':
            sub_level_preds = []
            sub_level_labels = []
            for j, sub in enumerate(self.subjects):
                subject_preds = []
                for i, (pred, inputObj) in enumerate(
                    zip(y_pred_keras, self.data)):

                    if inputObj.subject == str(sub[1]):
                        subject_preds.append(pred)
                        if len(sub_level_labels) == j:
                            sub_level_labels.append(
                                1 if inputObj.group == self.groups[1] else 0)
                sub_level_preds.append(np.mean(subject_preds))
            y_pred_keras = sub_level_preds
            test_set_labels = np.array(sub_level_labels)


        if plot_hist is True:
            from Plots import pred_hist
            pred_hist(
                y_pred_keras,
                test_set_labels,
                fname=None if fname is None
                    else checkpoint_dir+"/"+fname,
                group_names=self.group_names)

        return y_pred_keras

    def KfoldCrossVal(
        self,
        ML_function,
        normalize=None,
        learning_rate=0.001,
        lr_decay=False,
        beta1=0.9,
        beta2=0.999,
        epochs=100,
        iterations=1000,
        plot_ROC=False,
        k=1,
        plot_spec_avgs=False,
        C=1,
        kernel='linear',
        plot_PR=False,
        plot_Features=False,
        feat_select=False,
        num_feats=10,
            tt_split=0.33):
        """
        Resampling procedure used to evaluate ML models
        on a limited data sample

        Parameters:
            - k (default 5): int
                number of groups that a given data sample will be split into
        """
        for i in tqdm(range(k)):
            self.Prepare(
                k_fold=(i, k),
                normalize=normalize)

            if self.type == 'spectra' and plot_spec_avgs is True:
                from Standard import SpectralAverage
                specavgObj = SpectralAverage(self, training_only=True)
                specavgObj.plot(
                    fig_fname="specavg_"
                    + os.path.basename(self.trial_name)
                    + "_train_"
                    + str(i))

                specavgObj = SpectralAverage(self, testing_only=True)
                specavgObj.plot(
                    fig_fname="specavg_"
                    + os.path.basename(self.trial_name)
                    + "_test_"
                    + str(i))

            if ML_function == Classifier.CNN:
                model, y_pred, y_labels = ML_function(
                    learning_rate=learning_rate,
                    lr_decay=lr_decay,
                    beta1=beta1,
                    beta2=beta2,
                    epochs=epochs,
                    plot_ROC=plot_ROC)

                self.saveModelEvaluation('cnn', model, y_pred, y_labels, i)

            elif ML_function == Classifier.LDA:
                model, y_pred, y_labels = ML_function(
                    plot_data=False)

                self.saveModelEvaluation('lda', model, y_pred, y_labels, i)

            elif ML_function == Classifier.SVM:
                model, y_pred, y_labels = ML_function(
                    C=1,
                    kernel='linear',
                    iterations=1000,
                    plot_PR=False,
                    plot_Features=False,
                    feat_select=False,
                    num_feats=10)

                self.saveModelEvaluation('svm', model, y_pred, y_labels, i)

            elif ML_function == 'mixed':
                model, y_pred, y_labels = self.CNN(
                    learning_rate=learning_rate,
                    lr_decay=lr_decay,
                    beta1=beta1,
                    beta2=beta2,
                    epochs=epochs,
                    plot_ROC=plot_ROC)

                self.saveModelEvaluation('cnn', model, y_pred, y_labels, i)

                model, y_pred, y_labels = self.LDA(
                    plot_data=False)

                self.saveModelEvaluation('lda', model, y_pred, y_labels, i)

                model, y_pred, y_labels = self.SVM(
                    C=1,
                    kernel='linear',
                    iterations=1000,
                    plot_PR=False,
                    plot_Features=False,
                    feat_select=False,
                    num_feats=10)

                self.saveModelEvaluation('svm', model, y_pred, y_labels, i)

    def saveModelEvaluation(self, model_type, model, y_pred, y_labels, i):
        from Plots import roc

        f = open(
            self.type
            + "_"
            + model_type
            + "_"
            + os.path.basename(self.trial_name)
            + ".txt",
            'a')

        f.write("Subjects")
        for sub in self.subjects:
            f.write(str(sub[1]))
            f.write('\n')
        f.write('\n')

        auc = roc(y_pred, y_labels, plot=False)
        if auc < 0.5:
            auc = 1 - auc

        f.write('AUC')
        f.write('\n')
        f.write(str(auc))
        f.write('\n')
        f.write('Train ' + str(i))
        for sub in self.train_subjects:
            f.write(str(sub[1]))
            f.write('\n')
        f.write('\n')
        f.write('Test ' + str(i))
        for sub in self.test_subjects:
            f.write(str(sub[1]))
            f.write('\n')

        f.close()
