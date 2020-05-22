import config
import matplotlib.pyplot as plt
import numpy as np
import os

folders = [f for f in os.listdir(config.studyDirectory) if "contigs" in f]

fig, axs = plt.subplots(nrows=len(folders), ncols=len(config.network_channels), sharex=True, figsize=(12, 8))
cols = [str(channel) for channel in config.network_channels]
rows = [folder.rsplit('_', 1)[1] for folder in folders]

t = np.linspace(0, (config.contigLength / config.sampleRate), config.sampleRate, False)

i = 0
for folder in folders:
    source = config.studyDirectory+"/"+folder
    filtertype = folder.rsplit('_', 1)[1]
    arr = np.genfromtxt(source+"/"+config.plot_subject+"_"+config.plot_contig+".csv", delimiter=",")
    j = 0
    for sig in arr.T:
        axs[i][j].plot(t, sig)
        axs[i][j].axis([0, 1, np.amin(arr)-5, np.amax(arr)+5])
        j+=1
    i+=1

for ax, col in zip(axs[0], cols):
    ax.set_title(col)
for ax, row in zip(axs[:,0], rows):
    ax.set_ylabel(row + " filter", rotation=30, size="large")
for ax, col in zip(axs[-1], cols):
    ax.set_xlabel('Time [seconds]')

plt.tight_layout()
plt.show()
