import os
import shutil
import config
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal, stats


def BinarizeChannels(network_channels=config.network_channels):
    """
    Utility function to convert list of string channel names to a
    binary string corresponding to whether or not each default channel
    in config.channel_names is found in the input list

    Parameters:
        - network_channels: default config.network_channels

    Returns:
        - string of 0s and 1s
    """

    bin_str = ""

    for channel in config.channel_names:

        if channel in network_channels:

            bin_str = bin_str + "1"

        else:

            bin_str = bin_str + "0"

    return(bin_str)


def StringarizeChannels(bin_str):
    """
    Utility function to convert binary string channel names to a list of
    strings corresponding to whether or not each default channel
    in config.channel_names is found in the input list

    Parameters:
        - string of 0s and 1s

    Returns:
        - network_channels: default config.network_channels
    """

    channels = []
    for bin, chan in zip(bin_str, config.channel_names):
        if bin == "1":
            channels.append(chan)

    return channels


# just keeps subset of 19 channels as defined in network_channels
def FilterChannels(array, keep_channels, axisNum=1):
    """
    Returns a new array of input data containing only the channels
    provided in keep_channels; axisNum corresponds to the axis across
    which different channels are iterated

    Parameters:
        - array: (numpy.ndarray) input array
        - keep_channels: list of channel names to keep, should be a subset
          of config.channel_names
        - axisNum: (int) default 1

    Returns:
        - newarr: array with only certain channels kept in
    """

    filter_indeces = []

    # print("Old Shape of Dataset:", dataset.shape, "\n")
    # for each channel in my channel list
    for keep in keep_channels:

        filter_indeces.append(config.channel_names.index(keep))
        # get the index of that channel in the master channel list
        # add that index number to a new list filter_indeces

    newarr = np.take(array, filter_indeces, axisNum)

    return(newarr)


# takes one positional argument, path of TaskData folder
class TaskData:
    """
    Object used once data are cleaned and organized, in order to
    generate subsequent datatypes, such as Contigs or Spectra

    Parameters:
        - path: path to data directory (a task folder)
    """

    def __new__(self, path):

        if os.path.isdir(path):

            return super(TaskData, self).__new__(self)

        else:

            print("The path supplied is not a valid directory.")

            raise ValueError

    def __init__(self, path):

        self.path = path

        self.studyFolder = os.path.dirname(path)

        self.task = os.path.basename(self.path)

        self.task_fnames = os.listdir(self.path)

    def get_task_fnames(self, task):

        return(os.listdir(self.path))

    def set_subjects(self):

        self.subjects = set([
            fname[:config.participantNumLen]
            for fname in self.get_task_fnames(self.task)])

    # takes length (in samples @ 250 Hz / config.sample_rate)
    # as positional argument
    def gen_contigs(
        self,
        contigLength,
        network_channels=BinarizeChannels(config.network_channels),
        art_degree=0,
        erp=False,
        erp_degree=1,
            filter_band="nofilter"):

        """
        Generates Contig objects for every file possible in TaskData.path,
        appending each to TaskData.contigs

        Parameters:
            - contigLength: length in samples (@ 250 Hz or config.sample_rate)
            - network_channels: default config.network_channels
            - art_degree: (int) default 0, minimum value accepted to pass as a
              "clean" contig, when reading mask from .art file
            - erp: (bool) default False, if True then only contigs falling
              immediately after a "1" or a "2" in the corresponding .evt file
              will be accepted, i.e. only evoked responses
            - erp_degree: (int) default 1, lowest number in .evt which will be
              accepted as an erp event
        """

        if not hasattr(self, 'subjects'):

            self.set_subjects()

        if not hasattr(self, 'contigs'):

            self.contigs = []

        # make parent contigs folder
        if erp is False:
            if not os.path.isdir(self.studyFolder + "/contigs"):

                os.mkdir(self.studyFolder + "/contigs")

        elif erp is True:
            if not os.path.isdir(self.studyFolder + "/erps"):

                os.mkdir(self.studyFolder + "/erps")

        # make a child subdirectory called contigs_<task>_<contig_length>
        if erp is False:
            self.contigsFolder = self.studyFolder\
                                + "/contigs/"\
                                + self.task\
                                + "_"\
                                + str(contigLength)\
                                + "_"\
                                + network_channels\
                                + "_"\
                                + str(art_degree)

        elif erp is True:
            self.contigsFolder = self.studyFolder\
                                + "/erps/"\
                                + self.task\
                                + "_"\
                                + str(contigLength)\
                                + "_"\
                                + network_channels\
                                + "_"\
                                + str(art_degree)\
                                + "_"\
                                + str(erp_degree)

        os.mkdir(self.contigsFolder)
        # TODO: warning for pre-existing folder

        print("Contigifying Data:\n====================")

        for sub in tqdm(self.subjects):

            artfile = sub + "_" + self.task + ".art"

            # load in artifact file as np array
            # print("Artifact:"+self.path+"/"+artfile)
            artifact = np.genfromtxt(
                self.path + "/" + artfile,
                delimiter=" ")

            if erp is True:
                evtfile = sub + "_" + self.task + ".evt"
                events = np.genfromtxt(
                    self.path + "/" + evtfile,
                    delimiter=" ")

                if events.size == 0:
                    print(
                        "Most likely an empty text file was "
                        + "encountered. Skipping: " + evtfile)

                    continue

            if artifact.size == 0:

                print(
                    "Most likely an empty text file was "
                    + "encountered. Skipping: " + artfile)

                continue

            else:
                # get rid of channels we don't want the net to use
                artifact = FilterChannels(
                    artifact,
                    StringarizeChannels(network_channels),
                    1)

                # mask artifact array where numbers exceed art_degree
                mxi = np.ma.masked_where(artifact > art_degree, artifact)

                mxi = np.ma.filled(mxi.astype(float), np.nan)

                artifact = mxi

                indeces = []

                if erp is False:
                    # write list of start indexes for windows which meet
                    # contig requirements

                    i = 0

                    while i < artifact.shape[0] - contigLength:

                        stk = artifact[i:(i + contigLength), :]

                        if not np.any(np.isnan(stk)):

                            indeces.append(i)

                            i += contigLength

                        else:
                            # TODO
                            # contig alg can be sped up here to jump to
                            # last instance of NaN
                            i += 1

                else:
                    # only take oddball erp?
                    event_indeces = np.where(events >= erp_degree)[0]

                    for i in event_indeces:

                        stk = artifact[i:(i + contigLength), :]

                        if not np.any(np.isnan(stk)):

                            indeces.append(i)

                subfiles = [
                    fname for fname in self.task_fnames
                    if (sub == fname[:config.participantNumLen])
                    and ("eeg" in fname and "_" + filter_band in fname)]

                # subfiles = [fname for fname in subfiles if "eeg" in fname]

                j = 0

                for eegfile in subfiles:
                    # print("EEG file:"+self.path+"/"+eegfile)
                    data = np.genfromtxt(self.path+"/"+eegfile, delimiter=" ")

                    if data.size == 0:

                        print(
                            "Most likely an empty text file was "
                            + "encountered. Skipping: " + eegfile)

                    else:

                        band = eegfile.split('_')[2][:-4]

                        for i in indeces:

                            if erp is True:
                                self.contigs.append(
                                    Contig(
                                        data[i:(i + contigLength), :],
                                        i,
                                        sub,
                                        band,
                                        event=events[i]))

                            else:
                                self.contigs.append(
                                    Contig(
                                        data[i:(i + contigLength), :],
                                        i,
                                        sub,
                                        band))

    def write_contigs(self):
        """
        Writes TaskData.contigs objects to file, under TaskData.path / contigs
        """

        if not hasattr(self, 'contigs'):

            print("This instance of the class 'TaskData' has no\
                'contigs' attribute.")

            raise ValueError

        else:

            print("Writing Contigs:\n====================")

            for contig in tqdm(self.contigs):
                contig.write(
                    self.contigsFolder
                    + "/"
                    + contig.subject
                    + "_"
                    + contig.band
                    + "_"
                    + str(contig.startindex)
                    + ".csv")

    # takes length (in samples @ 250 Hz) as positional argument
    def gen_spectra(
        self,
        contigLength,
        network_channels=BinarizeChannels(config.network_channels),
        art_degree=0,
        erp=False,
        erp_degree=1,
            filter_band="nofilter"):

        """
        Generates Spectra objects for every file possible in TaskData.contigs,
        appending each to TaskData.spectra

        *** Note only reads contigs written to file currently

        Parameters:
            - contigLength: length in samples (@ 250 Hz or config.sample_rate)
            - network_channels: default config.network_channels
            - art_degree: (int) default 0, minimum value accepted to pass as a
              "clean" contig, when reading mask from .art file
        """

        if not hasattr(self, 'spectra'):

            self.spectra = []

        if erp is False:
            self.contigsFolder = self.studyFolder\
                + "/contigs/"\
                + self.task\
                + "_"\
                + str(contigLength)\
                + "_"\
                + network_channels\
                + "_"\
                + str(art_degree)

            self.spectraFolder = self.studyFolder\
                + "/spectra/"\
                + self.task\
                + "_"\
                + str(contigLength)\
                + "_"\
                + network_channels\
                + "_"\
                + str(art_degree)

        elif erp is True:
            self.contigsFolder = self.studyFolder\
                + "/erps/"\
                + self.task\
                + "_"\
                + str(contigLength)\
                + "_"\
                + network_channels\
                + "_"\
                + str(art_degree)\
                + "_"\
                + str(erp_degree)

            self.spectraFolder = self.studyFolder\
                + "/spectra/"\
                + self.task\
                + "_"\
                + str(contigLength)\
                + "_"\
                + network_channels\
                + "_"\
                + str(art_degree)\
                + "_"\
                + str(erp_degree)

        # make parent spectra folder
        if not os.path.isdir(self.studyFolder + "/spectra"):

            os.mkdir(self.studyFolder + "/spectra")

        # make a child subdirectory called\
        # <task>_<contig_length>_<binary_channels_code>
        # TODO: warning for pre-existing folder
        os.mkdir(self.spectraFolder)

        print("Fourier Transforming Data:\n====================")

        contigs = [
            fname for fname in os.listdir(self.contigsFolder)
            if "_" + filter_band in fname]
        for contig in tqdm(contigs):

            if "_" + filter_band in contig:

                temp = Contig(
                    np.genfromtxt(
                        self.contigsFolder + "/" + contig,
                        delimiter=","),
                    contig.split('_')[2][:-4],
                    contig[:config.participantNumLen],
                    contig.split('_')[1]).fft()
                if temp is not None:
                    self.spectra.append(temp)

    def write_spectra(self):
        """
        Writes self.spectra objects to file
        """

        if not hasattr(self, 'spectra'):

            print(
                "This instance of the class 'TaskData' has "
                + "no 'spectra' attribute.")

            raise ValueError

        else:

            print("Writing Spectra:\n====================")

            for spectrum in tqdm(self.spectra):

                spectrum.write(
                    self.spectraFolder
                    + "/"
                    + spectrum.subject
                    + "_"
                    + spectrum.band
                    + "_"
                    + str(spectrum.startindex)
                    + ".csv")


# takes 4 positional arguments:
# data array, sample of window start, subject ID,
# and frequency band or "nofilter"
class Contig:
    """
    Object used to store and denote information about a continuous \
    piece of EEG data. Often are generated using a strict (< 2) art_degree, \
    and so will be cleaner than raw data.

    Parameters:
        - data: (np.ndarray)
        - startindex: (int) timepoint in raw data that contig begins
        - subject: (str) subject code from which contig was generated
        - band: (str) frequency band from which contig was generated
        - source: path of original .eeg datafile
    """

    def __init__(
        self,
        data,
        startindex,
        subject,
        band,
        source=None,
            event=None):

        self.data = data

        self.startindex = startindex

        self.subject = subject

        self.band = band

        self.group = int(self.subject[0])

        self.source = source

        self.event = event

    def write(self, path):

        np.savetxt(
            path,
            self.data,
            delimiter=",",
            fmt="%2.1f")

    def fft(self):
        """
        Fourier transforms data

        Parameters: none

        Returns:
            - Spectra object
        """

        channel_number = 0

        for sig in self.data.T:

            electrode = config.network_channels[channel_number]

            # perform pwelch routine to extract PSD estimates by channel
            try:
                f, Pxx_den = signal.periodogram(
                    sig,
                    fs=float(config.sample_rate),
                    window='hann')
            except ValueError:
                print("Something went wrong processing the following contig:")
                print(self.subject, self.startindex, self.source)
                return None

            if channel_number == 0:

                spectral = np.array(f)

            spectral = np.column_stack((spectral, Pxx_den))

            channel_number += 1

        return(
            Spectra(
                spectral,
                spectral.T[0],
                self.startindex,
                self.subject,
                self.band))

    def plot(self):
        """
        Plots the contig
        """

        fig, axs = plt.subplots(nrows=1, figsize=(16, 8))

        trials = []

        plots = []

        for channel in config.network_channels:

            i = config.network_channels.index(channel)

            trial = self.data.T[i]

            trials.append(trial)

            wavs = trial * 0.1

            print(wavs)

            t = np.arange(0, len(wavs)) / config.sample_rate

            wavs = np.subtract(wavs, i * 25)

            plots.append(axs.plot(t, wavs)[0])

        axs.axis([
            t[0],
            t[-1],
            -25 * len(config.network_channels),
            5 * len(config.network_channels)])

        leg = axs.legend(config.network_channels, loc="right", fancybox=True)

        leg.get_frame().set_alpha(0.4)

        axs.grid(b=True)

        axs.set_title("Raw Waveforms: Subject " + self.subject)

        axs.set_ylabel("Voltage [mV]")

        axs.set_xlabel('Time [seconds]')

        # we will set up a dict mapping legend line to orig line, and enable
        # picking on the legend line
        lined = dict()

        for legline, origline in zip(leg.get_lines(), plots):

            legline.set_picker(5)  # 5 pts tolerance

            lined[legline] = origline

        def onpick(event):
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            legline = event.artist
            origline = lined[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            # Change the alpha on the line in the legend so
            # we can see what lines
            # have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', onpick)

        def movonpick(event):
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            movlegline = event.artist
            movorigline = movlined[movlegline]
            vis = not movorigline.get_visible()
            movorigline.set_visible(vis)
            # Change the alpha on the line in the legend so
            # we can see what lines
            # have been toggled
            if vis:
                movlegline.set_alpha(1.0)
            else:
                movlegline.set_alpha(0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', movonpick)

        plt.tight_layout()

        plt.show()


class Spectra:
    """
    Object used to store and denote information about a continuous \
    piece of EEG data's FFT.

    Parameters:
        - data: (np.ndarray)
        - freq: frequency of input data
        - startindex: (int) timepoint in raw data that contig begins
        - subject: (str) subject code from which contig was generated
        - band: (str) frequency band from which contig was generated
        - channels: list of channels included in predecessor Contig object
        - source: path of original .eeg datafile
    """

    def __init__(
        self,
        data,
        freq,
        startindex,
        subject,
        band="nofilter",
        channels=config.network_channels,
            source=None):

        self.data = data

        self.freq = freq

        self.startindex = startindex

        self.subject = subject

        self.band = band

        self.group = int(self.subject[0])

        self.channels = channels

        self.freq_res = (config.sample_rate / 2) / (len(self.freq) - 1)

        self.contig_length = config.sample_rate // self.freq_res

        self.source = source

    def write(self, path):

        np.savetxt(
            path,
            self.data,
            delimiter=",",
            fmt="%2.1f")
