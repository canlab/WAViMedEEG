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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from Prep import Contig, Spectra

# Utility function to move the midpoint of a colormap to be around
# the values of interest.
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# class ZscoreStandardize(normal):
#
#     def __init__(self, vmin=None, vmax=None, mean=None, clip=False):
#         self.mean = mean
#

# takes one positional argument: either "contigs" or "spectra"
class Classifier:
    def __init__(self, type, network_channels=config.network_channels):
        self.data = []
        self.type = type
        self.network_channels=config.network_channels
        self.trial_name = None

    def LoadData(self, path):
        fname = os.path.basename(path)
        self.trial_name = os.path.basename(os.path.dirname(os.path.dirname(path)))+"/"+os.path.basename(os.path.dirname(path))
        if self.type == "contigs":
            self.data.append(Contig(np.genfromtxt(path, delimiter=","), fname.split('_')[2][:-4], fname[:config.participantNumLen], fname.split('_')[1]))
        elif self.type == "spectra":
            self.data.append(Spectra(np.genfromtxt(path, delimiter=",").T[1:].T, np.genfromtxt(path, delimiter=",").T[0].T, fname.split('_')[2][:-4], fname[:config.participantNumLen], fname.split('_')[1], source=os.path.basename(os.path.dirname(path))))

    def Balance(self, parentPath):
        folders = ["ref 24-30", "ref 31-40", "ref 41-50", "ref 51-60", "ref 61-70", "ref 71-80", "ref 81+"]
        spectralFolder = self.trial_name
        totalpos = 0
        totalneg = 0

        for item in self.data:
            if item.group == 1:
                totalneg+=1
            elif item.group == 2:
                totalpos+=1

        fill = (totalpos - totalneg) / len(folders)
        for folder in tqdm(folders):
            fnames = os.listdir(parentPath+"/"+folder+"/"+spectralFolder)
            random.shuffle(fnames)
            i = 0
            while i < fill:
                self.LoadData(parentPath+"/"+folder+"/"+spectralFolder+"/"+fnames[i])
                i+=1

    def SVM(self, C=1, kernel=config.kernel_type, normalize=None, iterations=1000, plot_PR=False, plot_Features=False, feat_select=False, num_feats=10, lowbound=3, highbound=20, tt_split=0.33):
        # shuffle dataset
        random.shuffle(self.data)

        Features = np.array(self.data[0].freq[int(lowbound//self.data[0].freq_res)+1:int(highbound//self.data[0].freq_res)+2])

        # print("Shape of one data:", self.data[0].data.shape)

        # total num for each class
        totalpos = 0
        totalneg = 0

        for item in self.data:
            if item.group == 1:
                totalneg+=1
            elif item.group == 2:
                totalpos+=1

        print("Number of negative outcomes:", totalneg)
        print("Number of positive outcomes:", totalpos)


        # make list of subjects in this dataset
        subjects = list(set([(item.source, item.subject) for item in self.data]))


        # << old split methods >> #
                    # split data into train and test spectra (not split by subjects)
                    # x_train, x_test, y_train, y_test = train_test_split(dataset, labelmap, test_size=0.3)

                    # split data into train and test spectra (stratified by subjects)
                    # x_train, x_test, y_train, y_test = train_test_split(dataset, labelmap, test_size=0.3)


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
            # make label maps for each set: 1 = positive condition, -1 = negative condition
            train_dataset = np.stack([SpecObj.data[int(lowbound//SpecObj.freq_res)+1:int(highbound//SpecObj.freq_res)+2] for SpecObj in self.data if (SpecObj.source, SpecObj.subject) in train_subjects])
            train_labels = np.array([int(1) if (SpecObj.group==2) else int(-1) for SpecObj in self.data if (SpecObj.source, SpecObj.subject) in train_subjects])
            # print("Number of train examples:", len(train_dataset))


            test_dataset = np.stack([SpecObj.data[int(lowbound//SpecObj.freq_res)+1:int(highbound//SpecObj.freq_res)+2] for SpecObj in self.data if (SpecObj.source, SpecObj.subject) in test_subjects])
            test_labels = np.array([int(1) if (SpecObj.group==2) else int(-1) for SpecObj in self.data if (SpecObj.source, SpecObj.subject) in test_subjects])
            # print("Number of test examples:", len(test_dataset))

            # print("Frequency Resolution:", self.data[0].freq_res, "Hz")

        print("Number of samples in train:", train_dataset.shape[0])
        print("Number of samples in test:", test_dataset.shape[0])

        # reshape datasets
        nsamples, nx, ny = train_dataset.shape
        train_dataset = train_dataset.reshape((nsamples, nx*ny))

        nsamples, nx, ny = test_dataset.shape
        test_dataset = test_dataset.reshape((nsamples, nx*ny))


        # << Normalize features (z-score standardization) >> #
        # create classifier pipeline
        if normalize != None:
            if kernel == 'linear':
                clf = make_pipeline(myscaler, LinearSVC(C=C, random_state=0, max_iter=iterations, dual=False))
            elif kernel == 'rbf':
                clf = make_pipeline(myscaler, SVC(kernel='rbf', C=C, random_state=0, max_iter=iterations))
        else:
            if kernel == 'linear':
                clf = make_pipeline(LinearSVC(C=C, random_state=0, max_iter=iterations, dual=False))
            elif kernel == 'rbf':
                clf = make_pipeline(SVC(kernel='rbf', C=C, random_state=0, max_iter=iterations))

        # fit to train data
        clf.fit(train_dataset, train_labels)
        print('Classification accuracy on validation data: {:.3f}'.format(clf.score(test_dataset, test_labels)))

        # Compare to the weights of an SVM
        # only available using linear kernel
        if kernel == "linear":
            svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
            svm_weights /= svm_weights.sum()

        if plot_PR==True:
            # test_score = clf.decision_function(test_dataset)
            # average_precision = average_precision_score(test_dataset, test_score)

            # print('Average precision-recall score: {0:0.2f}'.format(average_precision))

            disp = plot_precision_recall_curve(clf, test_dataset, test_labels)
            disp.ax_.set_title('2-class Precision-Recall curve: ' 'AP={0:0.2f}'.format(average_precision))


        # univariate feature selection with F-test for feature scoring
        if feat_select==True:
            # We use the default selection function to select the ten
            # most significant features
            selector = SelectKBest(f_classif, k=num_feats)
            selector.fit(train_dataset, train_labels)
            scores = -np.log10(selector.pvalues_)
            scores /= scores.max()

            # Compare to the selected-weights of an SVM
            if normalize != None:
                if kernel == 'linear':
                    clf_selected = make_pipeline(SelectKBest(f_classif, k=num_feats), myscaler, LinearSVC(C=C, random_state=0, max_iter=iterations, dual=False))
                elif kernel == 'rbf':
                    clf_selected = make_pipeline(SelectKBest(f_classic, k=num_feats), myscaler, SVC(kernel='rbf', C=C, random_state=0, max_iter=iterations))
            else:
                if kernel == 'linear':
                    clf_selected = make_pipeline(SelectKBest(f_classif, k=num_feats), LinearSVC(C=C, random_state=0, max_iter=iterations, dual=False))
                elif kernel == 'rbf':
                    clf_selected = make_pipeline(SelectKbest(f_classic, k=num_feats), SVC(kernel='rbf', C=C, random_state=0, max_iter=iterations))

            clf_selected.fit(train_dataset, train_labels)
            print('Classification accuracy after univariate feature selection: {:.3f}'.format(clf_selected.score(test_dataset, test_labels)))

            # Compare to the weights of an SVM
            # only available using linear kernel
            if kernel == 'linear':
                svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
                svm_weights_selected /= svm_weights_selected.sum()

        if plot_Features==True:
            if kernel == 'linear':
                # set up figure and axes (rows)
                fig, axs = plt.subplots(nrows=len(self.network_channels), figsize=(20, 40))
                plt.rcParams['figure.dpi'] = 200

                # plt.clf()

                # X_indices = np.arange(train_dataset.shape[-1])
                X_indices = Features

                i = 0
                j = 0
                for channel in self.network_channels:
                    axs[j].set_title(channel)
                    axs[j].bar(X_indices - .25, svm_weights[i:i+len(X_indices)], width=.1, label='SVM weight')
                    if feat_select != True:
                        axs[j].legend()
                    i+=len(X_indices)
                    j+=1

                if feat_select==True:
                    i = 0
                    j = 0
                    for channel in self.network_channels:
                        axs[j].bar(X_indices - .45, scores[i:i+len(X_indices)], width=.1, label=r'Univariate score ($-Log(p_{value})$)')
                        # axs[j].bar(X_indices[selector.get_support()] - .05, svm_weights_selected[i:i+len(X_indices)], width=.1, label='SVM weights after selection')
                        axs[j].legend()
                        i+=len(X_indices)
                        j+=1

                axs[0].set_title("Comparing feature selection")
                axs[-1].set_xlabel('Feature number (in Hz)')
                # plt.yticks(())
                fig.tight_layout()
                plt.show()
