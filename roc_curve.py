import config
import os
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

# roc curve plot setup
rocfig, rocax = plt.subplots(1, 1, figsize=(10, 5))

# set list of colors for limited number of filters
if config.roc_type == "filter":
    resultsFolders = os.listdir(config.roc_source)
    resultsFolders = [config.roc_source+"/"+folder for folder in resultsFolders if "jacknife_evaluation" in folder]

    # colors = ['bo', 'rs', 'y1', 'k*', 'm+', 'gx']
    colors = ['b', 'r', 'y', 'k', 'm', 'g']

# color generator for shuffle / permuted label roc
if config.roc_type == "shuffle":
    resultsFolders = os.listdir(config.roc_source+"/jacknife_shuffle")
    resultsFolders = [config.roc_source+"/jacknife_shuffle/"+folder for folder in resultsFolders]
    resultsFolders.append(config.roc_source+"/jacknife_evaluation")

    NUM_COLORS = len(resultsFolders)
    cm = plt.get_cmap('gist_rainbow')
    rocax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

color = 0

def plot_pdf(pos_pdf, neg_pdf, ax):
    ax.fill(xax, pos_pdf, "b", alpha=0.5)
    ax.fill(xax, neg_pdf, "r", alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 5])
    ax.set_title("Probability Distribution: " + resultsFolder.replace(config.roc_source, ''))
    ax.set_ylabel('Counts')
    ax.set_xlabel('P(X="pain")')
    ax.legend(["pain", "control"])

def plot_roc(pos_pdf, neg_pdf, ax, color=None, marker=None, markersize=None):
    total_pos = np.sum(pos_pdf)
    total_neg = np.sum(neg_pdf)

    cum_TP = 0
    cum_FP = 0

    TPR_list = []
    FPR_list = []

    for i in range(len(xax)):
        if neg_pdf[i]>0:
            cum_TP+=pos_pdf[len(xax)-1-i]
            cum_FP+=neg_pdf[len(xax)-1-i]
        FPR = cum_FP / total_pos
        TPR = cum_TP / total_neg
        FPR_list.append(FPR)
        TPR_list.append(TPR)

    auc = np.sum(TPR_list)/len(xax)

    ax.plot(FPR_list, TPR_list, color=color, marker=marker, markersize=markersize)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_title("ROC Curve ")
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.grid()
    if config.roc_type == "filter":
        ax.legend([folder.replace(config.roc_source+"/", '') for folder in resultsFolders])

for resultsFolder in resultsFolders:
    fnames = [fname for fname in os.listdir(resultsFolder) if ".txt" in fname]
    subjects = [fname[:3] for fname in fnames]

    roc = []
    pos_predicts = []
    neg_predicts = []

    for fname in fnames:
        f = open(resultsFolder+"/"+fname, 'r')
        f.readline()
        f.readline()
        true_group = int(fname[0])
        prediction_group = float(f.readline().split()[1])
        if true_group == 1:
            pos_predicts.append(prediction_group)
        elif true_group == 2:
            neg_predicts.append(prediction_group)
        roc.append((true_group, prediction_group))

    pos_mu = np.mean(pos_predicts)
    pos_std = np.std(pos_predicts)
    neg_mu = np.mean(neg_predicts)
    neg_std = np.std(neg_predicts)

    xax = np.linspace(-0.5, 1.5, num=1000)

    pos_pdf = norm.pdf(xax, pos_mu, pos_std)
    neg_pdf = norm.pdf(xax, neg_mu, neg_std)

    if (config.roc_type == "shuffle") & (resultsFolder == config.roc_source+"/jacknife_evaluation"):
        plot_roc(pos_pdf, neg_pdf, rocax, color='r', marker='o', markersize=2)
    if (config.roc_type == "shuffle"):
        plot_roc(pos_pdf, neg_pdf, rocax)
    elif (config.roc_type == "filter"):
        plot_roc(pos_pdf, neg_pdf, rocax, colors[color])
    color+=1

    if config.roc_type == "filter":
        pdffig, pdfax = plt.subplots(1, 1, figsize=(10, 5))
        plot_pdf(pos_pdf, neg_pdf, pdfax)
        plt.hist(pos_predicts, bins=25, alpha=0.6, color='b')
        plt.hist(neg_predicts, bins=25, alpha=0.6, color='r')

rocax.plot(xax, xax, "--")

plt.show()
