from .. import Prep
from .. import config
from .. import Signals
import unittest
import random


class Test_Contigs(unittest.TestCase):
    def __init__(self):

        self.taskObj = Prep.TaskData('.')
        self.taskObj.studyFolder = None
        self.taskObj.task = None
        self.taskObj.task_fnames = None

        # make random list of fake subjects
        # adhering to subj-number lengths defined in config.py
        self.taskObj.subjects = []

        lo_bound = int('1' + '0'*(1-config.participantNumLen))
        hi_bound = int('2' + '9'*(1-config.participantNumLen))
        for i in range(10):
            num = None
            # no repeat subject numbers
            while num in self.taskObj.subjects:
                num = random.randint(1000, 2999)

            self.taskObj.subjects.append(num)


    def test_generation(self):

        
