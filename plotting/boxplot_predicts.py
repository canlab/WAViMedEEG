import numpy as np
import matplotlib.pyplot as plt
import config
import os

resultsFolders = [
    config.studyDirectory+"/results/flnk_1250_eyes_contigs",
    config.studyDirectory+"/results/p300_1250_eyes_contigs"
]

fig, axs = plt.subplots(ncols=len(resultsFolders), sharex=True, sharey=True)
axs[0].set_title('Predictions')

i = 0
for folder in resultsFolders:
    data = []
    fnames = os.listdir(folder)
    for fname in fnames:
        f = open(folder+"/"+fname, 'r')
        f.readline()
        f.readline()
        prediction = float(f.readline().split(' ')[1])
        data.append(prediction)
    axs[i].violinplot(data)
    i+=1
fig.show()
