import Prep
import config
import Signals
import unittest
import random
import numpy as np
import sys
import os
import shutil


class TestPreprocessing(unittest.TestCase):

    def test_taskdata_generation(self):

        for sub in taskObj.subjects:
            task_length = random.randint(2000, 5000)
            eeg_file = []
            art_file = []
            evt_file = []

            for chan in config.channel_names:
                eeg_file.append(
                    Signals.sin_wave(
                        task_length))
                art_file.append(
                    Signals.rand_bin_string(
                        task_length,
                        hi_bound=2,
                        weighted=True))

            eeg_file = np.stack(eeg_file).T
            art_file = np.stack(art_file).T

            evt_file = Signals.rand_bin_string(
                task_length,
                hi_bound=2,
                weighted=True).T

            np.savetxt(
                taskObj.studyFolder\
                +"/"+taskObj.task+"/"\
                +sub\
                +"_"\
                +taskObj.task\
                +"_nofilter.eeg",
                eeg_file,
                delimiter=" ",
                fmt="%2.1f")

            np.savetxt(
                taskObj.studyFolder\
                +"/"+taskObj.task+"/"\
                +sub\
                +"_"\
                +taskObj.task\
                +".art",
                art_file,
                delimiter=" ",
                fmt="%2.1f")

            np.savetxt(
                taskObj.studyFolder\
                +"/"+taskObj.task+"/"\
                +sub\
                +"_"\
                +taskObj.task\
                +".evt",
                evt_file,
                delimiter=" ",
                fmt="%2.1f")

            del eeg_file
            del art_file
            del evt_file

        self.assertEqual(
            len(os.listdir(taskObj.studyFolder+"/"+taskObj.task)),
            len(taskObj.subjects)*3)

    # def test_contig_generation(self):

        taskObj.task_fnames = taskObj.get_task_fnames(taskObj.task)

        length = random.randint(100, 2000)
        # art_degrees = [0, 1, 2]
        art_degrees = [1]
        # erps = [True, False]
        erps = [False]
        channels = ""
        for val in list(Signals.rand_bin_string(
            19,
            sample_rate=1)):
            channels += str(val)

        for art_degree in art_degrees:
            for erp in erps:
                taskObj.gen_contigs(
                    length,
                    network_channels=channels,
                    art_degree=art_degree,
                    erp=erp)
                taskObj.write_contigs()
                taskObj.gen_spectra(
                    length,
                    network_channels=channels,
                    art_degree=art_degree,
                    erp=erp)

        self.assertEqual(
            len(os.listdir(taskObj.studyFolder+"/contigs")),
            len(art_degrees))

        # self.assertEqual(
        #     len(os.listdir(taskObj.studyFolder+"/erps")),
        #     len(art_degrees))

        self.assertEqual(
            len(os.listdir(taskObj.studyFolder+"/spectra")),
            len(art_degrees)*len(erps))


if __name__ == '__main__':
    # unittest.TestCase.__init__(self,x)
    if not os.path.isdir('testdata/P300'):
        try:
            os.mkdir('testdata')
            os.mkdir('testdata/P300')
        except FileExistsError:
            try:
                os.mkdir('testdata/P300')
            except FileExistsError:
                print(
                    "Something went wrong creating test folders.",
                    "Try manually deleting ''/testdata/'")
                sys.exit(1)

    taskObj = Prep.TaskData(
        'testdata/'
        # +  str([
        #     task for task in config.tasks][
        #     random.randint(0, len(config.tasks)-1)]))
        + "P300")
    taskObj.task_fnames = None
    # else:
    #     shutil.rmtree('testdata/')
    #     os.mkdir('testdata/'+taskObj.task)

    # make random list of fake subjects
    # adhering to subj-number lengths defined in config.py
    taskObj.subjects = []

    lo_bound = int('1' + '0'*(1-config.participantNumLen))
    hi_bound = int('2' + '9'*(1-config.participantNumLen))
    for i in range(2):
        num = None
        num = random.randint(1000, 3999)
        # no repeat subject numbers
        while num in taskObj.subjects:
            num = random.randint(1000, 3999)

        taskObj.subjects.append(str(num))
    unittest.main()
