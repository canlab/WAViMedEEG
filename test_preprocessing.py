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

    def test_x(self):
        pass

    def test_raw_generation(self):
        task = [
            task for task in config.tasks][
            random.randint(0, len(config.tasks)-1)]
        for sub in taskObj.subjects:
            task_length = random.randint(1000, 10000)
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
                +"/"\
                +str(sub)\
                +"_"\
                +task\
                +"_nofilter.eeg",
                eeg_file,
                delimiter=" ",
                fmt="%2.1f")

            np.savetxt(
                taskObj.studyFolder\
                +"/"\
                +str(sub)\
                +"_"\
                +task\
                +".art",
                art_file,
                delimiter=" ",
                fmt="%2.1f")

            np.savetxt(
                taskObj.studyFolder\
                +"/"\
                +str(sub)\
                +"_"\
                +task\
                +".evt",
                evt_file,
                delimiter=" ",
                fmt="%2.1f")
        self.assertEqual(
            len(os.listdir(taskObj.studyFolder)),
            len(taskObj.subjects)*3)

    def test_contig_generation(self):
        length = random.randint(100, 20000)
        art_degrees = np.linspace(0, 2, 1, endpoint=True)
        erps = [True, False]

        for art_degree in art_degrees:
            for erp in erps:
                taskObj.gen_contigs(
                    length,
                    network_channels=Signals.rand_bin_string(
                        19,
                        sample_rate=1),
                    art_degree=art_degree,
                    erp=erp)

        assertEqual(
            len(os.listdir(taskObj.studyFolder+"/contigs")),
            len(art_degrees))

        assertEqual(
            len(os.listdir(taskObj.studyFolder+"/erps")),
            len(art_degrees))

        assertEqual(
            len(os.listdir(taskObj.studyFolder+"/spectra")),
            len(art_degrees)*len(erps))
            

if __name__ == '__main__':
    # unittest.TestCase.__init__(self,x)
    taskObj = Prep.TaskData('.')
    taskObj.studyFolder = 'testdata/raw'
    taskObj.task = None
    taskObj.task_fnames = None

    if not os.path.isdir(taskObj.studyFolder):
        try:
            os.mkdir('testdata')
        except FileExistsError:
            try:
                os.mkdir('testdata/raw')
            except FileExistsError:
                print(
                    "Something went wrong creating test folders.",
                    "Try manually deleting ''/testdata/raw'")
                sys.exit(1)
    else:
        shutil.rmtree('testdata/raw')
        os.mkdir('testdata/raw')


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

        taskObj.subjects.append(num)
    unittest.main()
