import Prep
import config
import Signals
import Standard
import ML
import Clean
import unittest
import random
import numpy as np
import sys
import os
import shutil


class TestPreprocessing(unittest.TestCase):

    def test_taskdata_generation(self):

        for sub in taskObj.subjects:
            task_length = random.randint(20, 200)
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
                taskObj.studyFolder
                + "/" + taskObj.task + "/"
                + sub
                + "_"
                + taskObj.task
                + "_nofilter.eeg",
                eeg_file,
                delimiter=" ",
                fmt="%2.1f")

            np.savetxt(
                taskObj.studyFolder
                + "/" + taskObj.task + "/"
                + sub
                + "_"
                + taskObj.task
                + ".art",
                art_file,
                delimiter=" ",
                fmt="%2.1f")

            np.savetxt(
                taskObj.studyFolder
                + "/" + taskObj.task + "/"
                + sub
                + "_"
                + taskObj.task
                + ".evt",
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

        length = random.randint(100, 750)
        art_degrees = [0, 1, 2]
        erps = [1, 2, None]
        taskObj.channels = ""
        for val in list(Signals.rand_bin_string(
            19,
                sample_rate=1)):
            taskObj.channels += str(val)

        for art_degree in art_degrees:
            for erp in erps:
                taskObj.gen_contigs(
                    length,
                    network_channels=taskObj.channels,
                    art_degree=art_degree,
                    erp_degree=erp)
                taskObj.write_contigs()
                taskObj.gen_spectra(
                    length,
                    network_channels=taskObj.channels,
                    art_degree=art_degree,
                    erp_degree=erp)
                taskObj.write_spectra()

        self.assertEqual(
            len(os.listdir(taskObj.studyFolder+"/contigs")),
            len(art_degrees))

        # self.assertEqual(
        #     len(os.listdir(taskObj.studyFolder+"/erps")),
        #     len(art_degrees))

        self.assertEqual(
            len(os.listdir(taskObj.studyFolder+"/spectra")),
            len(art_degrees)*len(erps))

        mytask = Standard.BandFilter(
            taskObj.studyFolder,
            "P300",
            type='highpass')
        mytask.gen_taskdata('alpha')
        mytask.write_taskdata()

        self.assertEqual(
            len([fname for fname in os.listdir(taskObj.studyFolder+"/P300")
                if "nofilter" in fname]),
            len([fname for fname in os.listdir(taskObj.studyFolder+"/P300")
                if "hialpha" in fname]))

        for data_type in ['spectra', 'contigs', 'erps']:
            options = os.listdir(taskObj.studyFolder+"/"+data_type+"/")
            patient_path = options[random.randint(0, len(options)-1)]
            patient_path = taskObj.studyFolder+"/"+data_type+"/"+patient_path

            # Instantiate a 'Classifier' Object
            myclf = ML.Classifier(data_type)

            # ============== Load Patient (Condition-Positive) Data ==========
            for fname in os.listdir(patient_path):
                if "_nofilter" in fname:
                    myclf.LoadData(patient_path+"/"+fname)

            myclf.Prepare(tt_split=0.33)

            myclf.CNN(epochs=10)


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

    else:
        shutil.rmtree('testdata/')
        os.mkdir('testdata/')
        os.mkdir('testdata/P300')

    taskObj = Prep.TaskData(
        'testdata/'
        # +  str([
        #     task for task in config.tasks][
        #     random.randint(0, len(config.tasks)-1)]))
        + "P300")
    taskObj.task_fnames = None

    # make random list of fake subjects
    # adhering to subj-number lengths defined in config.py
    taskObj.subjects = []

    lo_bound = int('1' + '0'*(config.participantNumLen-1))
    hi_bound = int('2' + '9'*(config.participantNumLen-1))
    for i in range(8):
        num = None
        num = random.randint(lo_bound, hi_bound)
        # no repeat subject numbers
        while num in taskObj.subjects:
            num = random.randint(lo_bound, hi_bound)

        taskObj.subjects.append(str(num))
    unittest.main()
