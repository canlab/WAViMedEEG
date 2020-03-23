import config
import os
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

resultsFolder = config.roc_source

fnames = [fname for fname in os.listdir(resultsFolder) if ".txt" in fname]
subjects = [fname[:3] for fname in fnames]

roc = []
positives = 0
negatives = 0

for fname in fnames:
    f = open(resultsFolder+"/"+fname, 'r')
    f.readline()
    f.readline()
    true_group = int(fname[0])
    prediction_group = float(f.readline().split()[1])
    roc.append((true_group, prediction_group))
    if fname[0] == "1":
        positives+=1
    elif fname[0] == "2":
        negatives+=1

x = []
y = []

i = 0
while i < 1:
    tp = 0
    tn = 0
    for sub in roc:
        if (sub[0] == 1) & (sub[1] > i):
            tp+=1
        if (sub[0] == 2) & (sub[1] < i):
            tn+=1
    tp_rate = tp / positives
    y.append(tp_rate)
    tn_rate = 1 - (tn / negatives)
    x.append(tn_rate)
    i += 0.001

pyplot.plot(x, y, scalex=True, scaley=True, data=None)
pyplot.show()
