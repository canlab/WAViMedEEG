import config
import matplotlib.pyplot as plt
import numpy as np
import os

spectral_folder = config.studyDirectory+"/spectral/"+config.selectedTask+"_"+str(config.contigLength)
fnames = [fname for fname in os.listdir(spectral_folder) if (config.plot_subject+"_"+config.plot_contig) in fname]
if len(fnames)<1:
    print("Are you sure that contig exists? I couldn't find it in the specified spectral folder.")
    quit()

fig, axs = plt.subplots(nrows=len(config.network_channels), ncols=1, sharex=True, figsize=(12, 8))
rows = [str(channel) for channel in config.network_channels]

t = np.linspace(0, (config.contigLength / config.sampleRate), config.sampleRate, False)

i = 0
for fname in fnames:
    arr = np.genfromtxt(spectral_folder+"/"+fname, delimiter=",")
    t = arr[0]
    sig = arr[1]
    axs[i].plot(t, sig)
    axs[i].axis([min(t), max(t), min(sig)-5, max(sig)+5])
    i+=1

axs[0].set_title("Power Spectral Density")
axs[-1].set_xlabel('Frequency [Hz]')
for ax, row in zip(axs, rows):
    ax.set_ylabel(row, rotation=30, size="large")


plt.tight_layout()
plt.show()
