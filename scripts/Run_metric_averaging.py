import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

metric_of_interest = sys.argv[1]

batchlog_folderpath = sys.argv[2]

# mainlog_folderpath = sys.argv[3]

batch_folders = [batchlog_folderpath + "/"+ folder\
    for folder in os.listdir(batchlog_folderpath)]

batch_metric = []

for i, folder in enumerate(batch_folders):

    df = pd.read_csv(folder+"/training.log")

    batch_metric.append(
        np.asarray(df[metric_of_interest])
    )

    if i == 0:
        epochs = np.asarray(df['epoch'])

# if metric_of_interest == 'loss':
batch_metric = [
metric / 40 for metric in batch_metric]

# mean_metric = np.mean([metric for metric in batch_metric], axis=0)

fig = plt.plot(dpi=100)

for metric in batch_metric:
    plt.plot(epochs, metric, alpha=1)

# plt.plot(epochs, mean_metric, label='mean', color='black')

plt.xlabel("Epochs")
plt.ylabel("Training Loss")

plt.legend(loc='lower right')

plt.tight_layout()

plt.show()
