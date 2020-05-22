import os
import numpy as np
import config
# import scipy
import random

from tqdm import tqdm
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

fnames = os.listdir(config.svm_source)
fnames_popper = fnames

Features = np.genfromtxt(config.svm_source+"/"+fnames[0], delimiter=",")[0][:5*64]
Labels = [0, 1]

dataset = []
targets = []

splt_char = "_"

pbar_len = len(fnames)
print("\nLoading spectra:")
print("=================\n")
pbar = tqdm(total=pbar_len)
while len(fnames_popper)>0:
    temp = fnames_popper[0].split(splt_char)
    contig_lead = splt_char.join(temp[:2])+"_"
    targets.append(int(contig_lead[0])-1)
    contig_fnames = [fname for fname in fnames_popper if contig_lead in fname]
    fnames_popper = [fname for fname in fnames_popper if fname not in contig_fnames]

    contig_spectrums = []
    for channel in config.network_channels:
        spectrum_fname = [fname for fname in contig_fnames if channel in fname]
        if len(spectrum_fname) != 1:
            print("Error: When reading spectrum filenames, more than one filename matched criteria for this channel:", channel)
            quit()
        spectrum_fname = spectrum_fname[0]
        spectrum = np.genfromtxt(config.svm_source+"/"+spectrum_fname, delimiter=",")[1]
        spectrum = spectrum[:5*64]
        contig_spectrums.append(spectrum)
        pbar.update(1)

    if len(contig_spectrums) != len(config.network_channels):
        print("Error: When reading spectrum files, there was a mismatch in the number of files read and the number of channels expected.")
        quit()

    contig_spectrums = np.stack(contig_spectrums)
    dataset.append(contig_spectrums)
pbar.close()

targetzero = [target for target in targets if target==0]
targetone = [target for target in targets if target==1]
print("Number of zeros:", len(targetzero))
print("Number of ones:", len(targetone))

shuffle_data_and_targets = list(zip(dataset, targets))
random.shuffle(shuffle_data_and_targets)

i = 0
j = 0


pbarlen = len(targetzero)-len(targetone)
print("Equalizing the size of your class data:")
print("=================\n")
pbar = tqdm(total=pbar_len)
while j < (len(targetzero)-len(targetone)):
    if shuffle_data_and_targets[i][1] == 0:
        shuffle_data_and_targets.pop(i)
        j+=1
        pbar.update(1)
    else:
        i+=1
pbar.close()

dataset, targets = zip(*shuffle_data_and_targets)

targetzero = [target for target in targets if target==0]
targetone = [target for target in targets if target==1]
print("Number of zeros:", len(targetzero))
print("Number of ones:", len(targetone))

dataset = np.stack(dataset)
nsamples, nx, ny = dataset.shape
dataset = dataset.reshape((nsamples, nx*ny))

# Standardize features
scaler = StandardScaler()
dataset_std = scaler.fit_transform(dataset)

print("Shape of dataset:", dataset.shape)
print("Length of targets:", len(targets))

print("\n")

x_train, x_test, y_train, y_test = train_test_split(dataset_std, targets, test_size=0.3)

# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

print("Training classifiers:")
print("=================\n")
C_range = np.logspace(-1, 3, 6)
# gamma_range = np.logspace(-3.5, -2.5, 12)
gamma_range = np.logspace(-6, -2, 6)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(dataset_std, targets)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-3, 1e-1, 1e1, 1e3]
gamma_2d_range = [1e-3, 1e-1, 1e1, 1e3]
classifiers = []

pbar_len = len(C_2d_range)*len(gamma_2d_range)
print("\nOptimizing SVM Parameters:")
print("=================\n")
pbar = tqdm(total=pbar_len)
for C in C_2d_range:
    for gamma in gamma_2d_range:

        #Create a svm Classifier
        clf = SVC(kernel=config.kernel_type, C=C, gamma=gamma)
        #Train the model using the training sets
        clf.fit(x_train, y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(x_test)
        # # Model Accuracy: how often is the classifier correct?
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        # # Model Precision: what percentage of positive tuples are labeled as such?
        # print("Precision:",metrics.precision_score(y_test, y_pred, zero_division=1))
        # # Model Recall: what percentage of positive tuples are labelled as such?
        # print("Recall:",metrics.recall_score(y_test, y_pred))

        classifiers.append((C, gamma, clf))
        pbar.update(1)
pbar.close()

# Visualization

if config.kernel_type == 'linear':
    # Plot data points and color using their class
    color = ['black' if c == 0 else 'lightgrey' for c in targets]
    plt.scatter(dataset_std[:,0], dataset_std[:,1], c=color)

    # Create the hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-2.5, 2.5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # Plot the hyperplane
    plt.plot(xx, yy)
elif config.kernel_type == 'poly':
    print("I'm a polynomial.")

elif config.kernel_type == 'rbf':
    # plt.figure(figsize=(8, 6))
    # xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    # for (k, (C, gamma, clf)) in enumerate(classifiers):
    #     # evaluate decision function in a grid
    #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #
    #     # visualize decision function for these parameters
    #     plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    #     plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
    #               size='medium')
    #
    #     # visualize parameter's effect on decision function
    #     plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    #     plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
    #                 edgecolors='k')
    #     plt.xticks(())
    #     plt.yticks(())
    #     plt.axis('tight')
    #
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')

plt.show()
