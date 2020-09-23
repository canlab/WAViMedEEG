import Prep, os
import config
import numpy as np
import matplotlib.pyplot as plt

# takes one positional argument, object from ML.Classifier
class SpectralAverage:
    def __new__(self, inputClf):
        if inputClf.type == "spectra":
            return super(SpectralAverage, self).__new__(self)
        else:
            print("This class requires an ML.Classifier object with type 'spectra' as a positional argument.")
            raise ValueError

    def __init__(self, inputClf):
        self.Clf = inputClf
        self.f = self.Clf.data[0].freq
        self.avgs = []
        self.groups = list(set(SpecObj.group for SpecObj in self.Clf.data))
        for group in self.groups:
            dataset = np.stack([np.sqrt(SpecObj.data) for SpecObj in self.Clf.data if SpecObj.group == group])
            self.avgs.append(np.mean(dataset, axis=0))

    def plot(self, channels=config.network_channels):
        plt.rcParams['figure.dpi'] = 200
        num_waves = len(self.avgs)
        num_channels = self.avgs[0].shape[-1]
        channels = channels

        fig, axs = plt.subplots(nrows=len(channels), sharex=False, figsize=(20, 40))
        fig.tight_layout()

        axs[-1].set_xlabel('Frequency')

        i = 0
        for channel in channels:
            j = 0
            for array in self.avgs:
                axs[i].plot(self.f[:100], array.T[i][:100], label=self.groups[j])
                j+=1
            axs[i].set_title(channel)
            axs[i].set_ylabel('Magnitude')
            axs[i].legend()
            i+=1

        plt.show()
