import config
import os
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

# resultsFolder = config.roc_source
resultsFolders = os.listdir(config.roc_source)
resultsFolders = [config.roc_source+"/"+folder for folder in resultsFolders if "jacknife" in folder]

# colors = ['bo', 'rs', 'y1', 'k*', 'm+', 'gx']
colors = ['b', 'r', 'y', 'k', 'm', 'g']


i = 0
for f in resultsFolders:
    print(f[19:] + " = " + colors[i])
    i+=1

color = 0

def pdf(x, std, mean):
    cons = 1.0 / np.sqrt(2*np.pi*(std**2))
    pdf_normal_dist = cons*np.exp(-((x-mean)**2)/(2.0*(std**2)))
    return pdf_normal_dist

def plot_pdf(pain_pdf, ctrl_pdf, ax):
    ax.fill(xax, pain_pdf, "b", alpha=0.5)
    ax.fill(xax, ctrl_pdf, "r", alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 5])
    ax.set_title("Probability Distribution: " + resultsFolder.replace(config.roc_source, ''))
    ax.set_ylabel('Counts')
    ax.set_xlabel('P(X="pain")')
    ax.legend(["pain", "control"])

def plot_roc(pain_pdf, ctrl_pdf, ax, color):
    total_pain = np.sum(pain_pdf)
    total_ctrl = np.sum(ctrl_pdf)

    cum_TP = 0
    cum_FP = 0

    TPR_list = []
    FPR_list = []

    for i in range(len(xax)):
        if ctrl_pdf[i]>0:
            cum_TP+=pain_pdf[len(xax)-1-i]
            cum_FP+=ctrl_pdf[len(xax)-1-i]
        FPR = cum_FP / total_pain
        TPR = cum_TP / total_ctrl
        FPR_list.append(FPR)
        TPR_list.append(TPR)

    auc = np.sum(TPR_list)/len(xax)

    ax.plot(FPR_list, TPR_list, color)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_title("ROC Curve ")
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.grid()
    ax.legend([folder.replace(config.roc_source+"/", '') for folder in resultsFolders])

# roc curve plot setup
rocfig, rocax = plt.subplots(1, 1, figsize=(10, 5))

for resultsFolder in resultsFolders:
    fnames = [fname for fname in os.listdir(resultsFolder) if ".txt" in fname]
    subjects = [fname[:3] for fname in fnames]

    roc = []
    positives = 0
    negatives = 0
    pain_predicts = []
    ctrl_predicts = []

    for fname in fnames:
        f = open(resultsFolder+"/"+fname, 'r')
        f.readline()
        f.readline()
        true_group = int(fname[0])
        prediction_group = float(f.readline().split()[1])
        if true_group == 1:
            pain_predicts.append(prediction_group)
        elif true_group == 2:
            ctrl_predicts.append(prediction_group)
        roc.append((true_group, prediction_group))
        if fname[0] == "1":
            positives+=1
        elif fname[0] == "2":
            negatives+=1

    # x = []
    # y = []
    print("Input for ", resultsFolder.replace(config.roc_source, ''))
    print("Pain:", pain_predicts)
    print("Ctrl:", ctrl_predicts)

    pain_predicts.sort()
    ctrl_predicts.sort()

    pain_mu = np.mean(pain_predicts)
    pain_std = np.std(pain_predicts)
    ctrl_mu = np.mean(ctrl_predicts)
    ctrl_std = np.std(ctrl_predicts)

    xax = np.linspace(-0.5, 1.5, num=1000)

    pain_pdf = norm.pdf(xax, pain_mu, pain_std)
    ctrl_pdf = norm.pdf(xax, ctrl_mu, ctrl_std)
    # pain_pdf = pdf(xax, pain_std, pain_mu)
    # ctrl_pdf = pdf(xax, ctrl_std, ctrl_mu)

    plot_roc(pain_pdf, ctrl_pdf, rocax, colors[color])
    color+=1

    pdffig, pdfax = plt.subplots(1, 1, figsize=(10, 5))
    plot_pdf(pain_pdf, ctrl_pdf, pdfax)
    plt.hist(pain_predicts, bins=25, alpha=0.6, color='b')
    plt.hist(ctrl_predicts, bins=25, alpha=0.6, color='r')

    # i = 0
    # while i < 1:
    #     tp = 0
    #     tn = 0
    #     for sub in roc:
    #         if (sub[0] == 1) & (sub[1] > i):
    #             tp+=1
    #         if (sub[0] == 2) & (sub[1] < i):
    #             tn+=1
    #     tp_rate = tp / positives
    #     y.append(tp_rate)
    #     tn_rate = 1 - (tn / negatives)
    #     x.append(tn_rate)
    #     i += 0.001


rocax.plot(xax, xax, "--")

plt.show()
