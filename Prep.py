import os, shutil
import config
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy
from scipy import signal, stats

def BinarizeChannels(network_channels=config.network_channels):
    bin_str = ""
    for channel in config.channel_names:
        if channel in network_channels:
            bin_str = bin_str + "1"
        else:
            bin_str = bin_str + "0"
    return(bin_str)

# just keeps subset of 19 channels as defined in network_channels
def FilterChannels(array, keep_channels, axisNum):
    filter_indeces = []
    # print("Old Shape of Dataset:", dataset.shape, "\n")
    # for each channel in my channel list
    for keep in keep_channels:
        filter_indeces.append(config.channel_names.index(keep))
    #   get the index of that channel in the master channel list
    #   add that index number to a new list filter_indeces

    newarr = np.take(array, filter_indeces, axisNum)
    # print("New Shape of Dataset:", filtered_dataset.shape, "\n")
    return(newarr)

def generate_sparse_indexes(art, length):
    i = 0
    startindexes = []
    while i < art.shape[0]-length:
        stk = art[i:(i+length),:]
        if not np.any(np.isnan(stk)):
            startindexes.append(i)
            i+=length
        else:
            i+=1
    return startindexes

# takes one positional argument, path of study folder
class StudyFolder:

    def __new__(self, path):
        if os.path.isdir(path):
            return super(StudyFolder, self).__new__(self)
        else:
            print("The path supplied is not a valid directory.")
            raise ValueError

    def __init__(self, path):
        self.path = path
        print("\nInitializing New Study Directory at: " + self.path)
        self.raw_fnames = os.listdir(self.path+"/raw")

    # standardizes task names / file structure and anonymizes subject headers
    # translator stored in StudyFolder/task_translator.txt
    def autoclean(self):
        for task in config.tasks:
            for irregular in config.tasks[task]:
                self.standardize(irregular, task)
            if os.path.isdir(self.path+"/"+task):
                self.anon(task)
        if len(os.listdir(self.path)) != 0:
            print("Some raw files couldn't be automatically standardized. You should review them in /raw before moving forward with analysis.")

    def set_raw_fnames(self):
        self.raw_fnames = os.listdir(self.path+"/raw")

    def get_task_fnames(self, task):
        return(os.listdir(self.path+"/"+task))

    # standardizes every filename possible, using alternative (unclean)
    # fnames from the WAVi desktop which are written in the tasks dict in config.py
    def standardize(self, old, new):
        if not os.path.isdir(self.path+"/"+new):
            print("Making new task folder: ", new)
            os.mkdir(self.path+"/"+new)

        for fname in [fname for fname in self.raw_fnames if old in fname]:
            newfname = fname.replace(old, new)
            shutil.move(self.path+"/raw/"+fname, self.path+"/"+new+"/"+newfname)

        self.set_raw_fnames()
        if len(os.listdir(self.path+"/"+new))==0:
            os.rmdir(self.path+"/"+new)

    # anonymizes sets of standardized task data which can then be read into a Trials object
    def anon(self, task, groupNum=1):
        translator = {}
        subject_leads = set([fname.replace(task, '')[:-4] for fname in self.get_task_fnames(task)])
        i = 0
        f = open(self.path+"/translator_"+task+".txt", "w")
        for lead in subject_leads:
            translator[lead] = str(groupNum)+"0"*(config.participantNumLen-len(str(i))-1)+str(i)
            i+=1
            f.write(lead)
            f.write("\t")
            f.write(translator[lead])
            f.write("\n")
            files = [fname for fname in self.get_task_fnames(task) if lead in fname]
            for file in files:
                newfile = file.replace(lead, translator[lead]+"_")
                shutil.move(self.path+"/"+task+"/"+file, self.path+"/"+task+"/"+newfile)


# takes one positional argument, path of trials folder
class Trials:

    def __new__(self, path):
        if os.path.isdir(path):
            return super(Trials, self).__new__(self)
        else:
            print("The path supplied is not a valid directory.")
            raise ValueError

    def __init__(self, path):
        self.path = path
        self.studyFolder = os.path.dirname(path)
        self.task = os.path.basename(self.path)
        self.task_fnames = os.listdir(self.path)
        self.network_channels = config.network_channels

    def get_task_fnames(self, task):
        return(os.listdir(self.path))

    def no_filter_rename(self):
        for fname in self.task_fnames:
            shutil.move(self.path+"/"+fname, self.path+"/"+fname[:-4]+"_nofilter"+fname[-4:])
        self.task_fnames = self.get_task_fnames(self.task)

    def set_subjects(self):
        self.subjects = set([fname[:config.participantNumLen] for fname in self.get_task_fnames(self.task)])

    # takes length (in samples @ 250 Hz / config.sampleRate) as positional argument
    def gen_contigs(self, contigLength, network_channels=config.network_channels, artDegree=config.artDegree):

        if not hasattr(self, 'subjects'):
            self.set_subjects()

        if not hasattr(self, 'contigs'):
            self.contigs = []

        # make parent contigs folder
        if not os.path.isdir(self.studyFolder+"/contigs"):
            os.mkdir(self.studyFolder+"/contigs")

        # make a child subdirectory called contigs_<task>_<contig_length>
        try:
            self.contigsFolder = self.studyFolder+"/contigs/"+self.task+"_"+str(contigLength)+"_"+BinarizeChannels(network_channels=self.network_channels)+"_"+str(artDegree)
            os.mkdir(self.contigsFolder)
        except:
            print("Couldn't create the specified contigs folder.\n")

        print("Contigifying Data:\n====================")

        for sub in tqdm(self.subjects):
            artfile = sub+"_"+self.task+"_nofilter.art"

            # load in artifact file as np array
            # print("Artifact:"+self.path+"/"+artfile)
            artifact = np.genfromtxt(self.path+"/"+artfile, delimiter=" ")

            if artifact.size == 0:
                print("Most likely an empty text file was encountered. Skipping: " + artfile)
                continue

            else:
                # get rid of channels we don't want the net to use
                artifact = FilterChannels(artifact, network_channels, 1)
                # mask artifact array where numbers exceed artDegree
                mxi = np.ma.masked_where(artifact > artDegree, artifact)
                mxi = np.ma.filled(mxi.astype(float), np.nan)
                artifact = mxi

                # write list of start indexes for windows which meet
                # contig requirements
                i = 0
                indeces = []
                while i < artifact.shape[0]-contigLength:
                    stk = artifact[i:(i+contigLength),:]
                    if not np.any(np.isnan(stk)):
                        indeces.append(i)
                        i+=contigLength
                    else:
                        i+=1

                subfiles = [fname for fname in self.task_fnames if (sub==fname[:config.participantNumLen] and "eeg" in fname)]
                # subfiles = [fname for fname in subfiles if "eeg" in fname]

                j = 0
                for eegfile in subfiles:
                    # print("EEG file:"+self.path+"/"+eegfile)
                    data = np.genfromtxt(self.path+"/"+eegfile, delimiter=" ")
                    if data.size == 0:
                        print("Most likely an empty text file was encountered. Skipping: " + eegfile)

                    else:
                        band = eegfile.split('_')[2][:-4]
                        for i in indeces:
                            self.contigs.append(Contig(data[i:(i+contigLength),:], i, sub, band))

    def write_contigs(self):
        if not hasattr(self, 'contigs'):
            print("This instance of the class 'Trials' has no 'contigs' attribute.")
            raise ValueError
        else:
            print("Writing Contigs:\n====================")
            for contig in tqdm(self.contigs):
                contig.write(self.contigsFolder+"/"+contig.subject+"_"+contig.band+"_"+str(contig.startindex)+".csv")

    # takes length (in samples @ 250 Hz) as positional argument
    def gen_spectra(self, contigLength, network_channels=config.network_channels, artDegree=config.artDegree):
        if not hasattr(self, 'spectra'):
            self.spectra = []
        self.contigsFolder = self.studyFolder+"/contigs/"+self.task+"_"+str(contigLength)+"_"+BinarizeChannels(network_channels=self.network_channels)+"_"+str(artDegree)
        self.spectraFolder = self.studyFolder+"/spectra/"+self.task+"_"+str(contigLength)+"_"+BinarizeChannels(network_channels=self.network_channels)+"_"+str(artDegree)
        # make parent spectra folder
        if not os.path.isdir(self.studyFolder+"/spectra"):
            os.mkdir(self.studyFolder+"/spectra")

        # make a child subdirectory called <task>_<contig_length>_<binary_channels_code>
        try:
            self.spectraFolder = self.studyFolder+"/spectra/"+self.task+"_"+str(contigLength)+"_"+BinarizeChannels(network_channels=self.network_channels)+"_"+str(artDegree)
            os.mkdir(self.spectraFolder)
        except:
            print("Couldn't create the specified spectra folder.\n")

        print("Fourier Transforming Data:\n====================")
        for contig in tqdm(os.listdir(self.contigsFolder)):
            self.spectra.append(Contig(np.genfromtxt(self.contigsFolder+"/"+contig, delimiter=","), contig.split('_')[2][:-4], contig[:config.participantNumLen], contig.split('_')[1]).fft())

    def write_spectra(self):
        if not hasattr(self, 'spectra'):
            print("This instance of the class 'Trials' has no 'spectra' attribute.")
            raise ValueError
        else:
            print("Writing Spectra:\n====================")
            for spectrum in tqdm(self.spectra):
                spectrum.write(self.spectraFolder+"/"+spectrum.subject+"_"+spectrum.band+"_"+str(spectrum.startindex)+".csv")

# takes 4 positional arguments: data array, sample of window start, subject ID, and frequency band or "nofilter"
class Contig:
    def __init__(self, data, startindex, subject, band, source=None):
        self.data = data
        self.startindex = startindex
        self.subject = subject
        self.band = band
        self.group = self.subject[0]
        self.source = source

    def write(self, path):
        np.savetxt(path, self.data, delimiter=",", fmt="%2.1f")

    def fft(self):
        channel_number = 0
        # spectral = np.ndarray(shape=(len(self.data), len(config.channel_names))).T
        for sig in self.data.T:
            electrode = config.network_channels[channel_number]
            # perform pwelch routine to extract PSD estimates by channel
            f, Pxx_den = scipy.signal.periodogram(
                sig,
                fs=float(config.sampleRate),
                window='hann'
                )
            if channel_number == 0:
                spectral = np.array(f)
            spectral = np.column_stack((spectral, Pxx_den))
            channel_number+=1
        return(Spectra(spectral, spectral.T[0], self.startindex, self.subject, self.band))

    def plot(self):
        fig, axs = plt.subplots(nrows=1, figsize=(16, 8))

        trials = []
        plots = []
        for channel in config.network_channels:
            i = config.network_channels.index(channel)
            trial = self.data.T[i]
            trials.append(trial)
            wavs = trial * 0.1
            print(wavs)
            t = np.arange(0, len(wavs)) / config.sampleRate
            wavs = np.subtract(wavs, i*25)
            plots.append(axs.plot(t, wavs)[0])

        axs.axis([t[0], t[-1], -25*len(config.network_channels), 5*len(config.network_channels)])
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
            # Change the alpha on the line in the legend so we can see what lines
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
            # Change the alpha on the line in the legend so we can see what lines
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
    def __init__(self, data, freq, startindex, subject, band="nofilter", channels=config.network_channels, source=None):
        self.data = data
        self.freq = freq
        self.startindex = startindex
        self.subject = subject
        self.band = band
        self.group = int(self.subject[0])
        self.channels = channels
        self.freq_res = (config.sampleRate/2) / (len(self.freq)-1)
        self.contig_length = config.sampleRate // self.freq_res
        self.source = source

    def write(self, path):
        np.savetxt(path, self.data, delimiter=",", fmt="%2.1f")
