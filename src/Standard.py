from src import Prep
from src import config
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
from tqdm import tqdm
import scipy.signal
from scipy.signal import butter, lfilter


# takes one positional argument, object from ML.Classifier
class SpectralAverage:
    def __new__(
        self,
        inputClf,
        lowbound=0,
        highbound=25,
        training_only=False,
            testing_only=False):

        if inputClf.type == "spectra":
            return super(SpectralAverage, self).__new__(self)

        else:
            print("This class requires an ML.Classifier object with\
                  type 'spectra' as a positional argument.")
            raise ValueError

    def __init__(
        self,
        inputClf,
        use_gpu=False,
        lowbound=0,
        highbound=25,
        training_only=False,
            testing_only=False):

        if use_gpu is True:
            import cupy as np

        self.Clf = inputClf

        self.lowbound = lowbound

        self.highbound = highbound

        self.f = np.array(self.Clf.data[0].freq[
            int(self.lowbound // self.Clf.data[0].freq_res):
            int(self.highbound // self.Clf.data[0].freq_res) + 1])

        self.avgs = []

        self.groups = list(set(
            [SpecObj.group for SpecObj in self.Clf.data]))

        for group in self.groups:
            if (training_only is False) and (testing_only is False):
                dataset = np.stack([
                    np.sqrt(SpecObj.data[
                        int(self.lowbound // SpecObj.freq_res):
                        int(self.highbound // SpecObj.freq_res) + 1])
                    for SpecObj in self.Clf.data
                    if SpecObj.group == group])

            elif training_only is True:
                dataset = np.stack([
                    np.sqrt(SpecObj.data[
                        int(self.lowbound // SpecObj.freq_res):
                        int(self.highbound // SpecObj.freq_res) + 1])
                    for SpecObj in self.Clf.data
                    if SpecObj.group == group
                    and str(SpecObj.subject) in
                    [sub[1] for sub in self.Clf.train_subjects]])

            elif testing_only is True:
                dataset = np.stack([
                    np.sqrt(SpecObj.data[
                        int(self.lowbound // SpecObj.freq_res):
                        int(self.highbound // SpecObj.freq_res) + 1])
                    for SpecObj in self.Clf.data
                    if SpecObj.group == group
                    and str(SpecObj.subject) in
                    [sub[1] for sub in self.Clf.test_subjects]])

            self.avgs.append(np.mean(dataset, axis=0))

    def plot(self, channels=config.network_channels, fig_fname=None):

        plt.rcParams['figure.dpi'] = 100

        num_waves = len(self.avgs)

        num_channels = self.avgs[0].shape[-1]

        channels = channels

        fig, axs = plt.subplots(
            nrows=len(channels),
            sharex=False,
            figsize=(12, 1*num_channels))

        fig.tight_layout()

        axs[-1].set_xlabel('Frequency')

        i = 0

        for channel in channels:

            j = 0

            for array in self.avgs:

                axs[i].plot(
                    self.f,
                    array.T[i],
                    label=config.group_names[self.groups[j]],
                    color=config.group_colors[self.groups[j]])

                j += 1

            axs[i].set_title(channel)

            axs[i].set_ylabel('Magnitude')

            axs[i].legend()

            axs[i].set_xticks(
                np.linspace(
                    self.f[0],
                    self.f[-1],
                    len(self.f),
                    endpoint=True))

            i += 1

        if fig_fname is None:
            plt.show()

        if fig_fname is not None:
            fig.suptitle(fig_fname)
            fig.savefig(fig_fname)
            plt.close(fig)


class BandFilter:
    # removes a given frequency band's power from task data trials

    def __init__(self, study_folder, task, type="bandstop", use_gpu=False):

        self.study_folder = study_folder
        self.task = task
        self.type = type
        self.new_data = []

        if use_gpu is True:
            import cupy as np

    def gen_taskdata(self, filter_band):

        range = config.frequency_bands[filter_band]

        if self.type == 'lowpass':
            sos = scipy.signal.butter(
                4,
                range[0],
                btype=self.type,
                fs=config.sample_rate,
                output='sos')
        elif self.type == 'highpass':
            sos = scipy.signal.butter(
                4,
                range[1],
                btype=self.type,
                fs=config.sample_rate,
                output='sos')
        else:
            sos = scipy.signal.butter(
                4,
                [range[0],
                    range[1]],
                btype=self.type,
                fs=config.sample_rate,
                output='sos')

        fnames = [
            fname for fname in os.listdir(self.study_folder+"/"+self.task)
            if "_nofilter" in fname]

        print("Generating filtered trials:")
        for fname in tqdm(fnames):

            try:
                arr = np.genfromtxt(
                    self.study_folder+"/"+self.task+"/"+fname,
                    delimiter=" ")
            except Exception:
                print(fname, " FAILED")
                print("Couldn't load data. Needs delim fix probably.")
                sys.exit(3)

            if arr.size == 0:
                print(
                    "Most likely an empty text file was "
                    + "encountered. Skipping: " + fname)
                continue

            post = []

            for sig in arr.T:
                filtered = scipy.signal.sosfilt(sos, sig)
                post.append(filtered)

            post = np.stack(post)

            if self.type == "bandstop":
                new_fname = fname.replace("nofilter", "no"+filter_band)
            elif self.type == "bandpass":
                new_fname = fname.replace("nofilter", filter_band)
            elif self.type == "lowpass":
                new_fname = fname.replace("nofilter", "lo"+filter_band)
            elif self.type == "highpass":
                new_fname = fname.replace("nofilter", "hi"+filter_band)

            self.new_data.append((new_fname, post.T))

    def write_taskdata(self):

        print("Writing filtered trials:")
        for file in tqdm(self.new_data):

            np.savetxt(
                self.study_folder+"/"+self.task+"/"+file[0], file[1],
                delimiter=" ", fmt="%2.1f")


class CoherenceMap:
    def __init__(
        self,
        array=None,
        channel_names=config.channel_names,
        source=None,
        subject=None,
        task=None,
        length=None,
        sample_rate=config.sample_rate):

        if array is not None:
            self.array = array.T
        else:
            self.array = None
        self.channel_names = channel_names
        self.source = source
        self.subject = subject
        self.task = task
        if self.array is not None:
            self.length = self.array.shape[1]
        else:
            self.length = 0
        self.sample_rate = sample_rate

        if self.array is not None:
            edgelist = []
            cohs = []
            # channels i=0:-1
            for i, (sig1, chan1) in enumerate(
                zip(self.array, self.channel_names)):
                if i > len(self.array):
                    continue
                # channels i+1:
                for j, (sig2, chan2) in enumerate(
                    zip(self.array, self.channel_names)):
                    if j <= i:
                        continue
                    f, Cxy = scipy.signal.coherence(
                        sig1,
                        sig2,
                        fs=self.sample_rate)
                    # on first iteration, element 'f' of dict is frequency list
                    if j == 1:
                        labels.append('f')
                        cohs.append(f)

                    labels.append(str(channel_names[i] + '-' + channel_names[j]))
                    cohs.append(Cxy)

            labels = np.array(labels)
            cohs = np.array(cohs)
            self.map = (labels, cohs)

    def plot_coherences(self):
        f = self.map[1][0]
        cohs = self.map[1][1:10]
        labels = self.map[0][1:10]

        fig, axs = plt.subplots(nrows=len(labels), sharex=True)
        for label, Cxy, ax in zip(labels, cohs, axs):
            ax.semilogy(f, Cxy)
            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('Coherence')
            ax.set_title(str(label))
        # plt.tight_layout()
        plt.show()

    def write(self, path):
        np.savetxt(
            path,
            self.map[1],
            delimiter=",",
            fmt="%2.1f")
