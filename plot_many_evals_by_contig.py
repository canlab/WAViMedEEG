import config
import os
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

studyFolders = [folder for folder in os.listdir(config.roc_sourceDir_many) if config.roc_source_keyword_many in folder]
resultsFolders = [config.roc_sourceDir_many+"/"+folder+config.req_many_eval_path for folder in studyFolders]
resultsFolders = [folder for folder in resultsFolders if os.path.isdir(folder)]
resultsFolders = sorted(resultsFolders)

# PDF plot setup
pdffig, pdfax = plt.subplots(nrows=len(resultsFolders), ncols=1, figsize=(16, 8))

# set list of colors
print(resultsFolders)
# colors = ['bo', 'rs', 'y1', 'k*', 'm+', 'gx']
colors = ['b', 'r', 'y', 'k', 'm', 'g']

color = 0

def plot_pdf(pos_pdf, neg_pdf, ax, foldername, color):
    ax.fill(xax, pos_pdf, 'c', alpha=0.5)
    ax.fill(xax, neg_pdf, color, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 50])
    ax.set_title("PDF: " + foldername.replace(config.roc_sourceDir_many, "").replace(config.req_many_eval_path, ""))
    ax.set_ylabel('Counts')
    ax.set_xlabel('P(X="pain")')
    ax.legend(["pain", "control"])

for resultsFolder in resultsFolders:
    fnames = [fname for fname in os.listdir(resultsFolder) if ".txt" in fname]
    subjects = [fname[:3] for fname in fnames]

    pos_predicts = []
    neg_predicts = []

    for fname in fnames:
        f = open(resultsFolder+"/"+fname, 'r')
        f.readline()
        true_group = int(fname[0])
        prediction_group = float(f.readline().split()[2])
        if true_group == 1:
            pos_predicts.append(prediction_group)
        elif true_group == 2:
            neg_predicts.append(prediction_group)

    pos_mu = np.mean(pos_predicts)
    pos_std = np.std(pos_predicts)
    neg_mu = np.mean(neg_predicts)
    neg_std = np.std(neg_predicts)

    xax = np.linspace(-0.5, 1.5, num=1000)

    pos_pdf = norm.pdf(xax, pos_mu, pos_std)
    neg_pdf = norm.pdf(xax, neg_mu, neg_std)

    plot_pdf(pos_pdf, neg_pdf, pdfax[color], resultsFolder, colors[color])
    pdfax[color].hist(pos_predicts, bins=25, alpha=0.6, color="c")
    pdfax[color].hist(neg_predicts, bins=25, alpha=0.6, color=colors[color])
    color+=1

plt.show()
