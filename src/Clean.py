from src import config
import os
import shutil
from tqdm import tqdm


def is_anonymous(fname, task):
    """
    Check to determine if a given filename is already anonymized,
    in accordance with the current filename convention and task name.

    Parameters:
        - fname: filename to be checked
        - task: task name that should be used in check
    """
    fname = fname[:-4]
    try:
        subject_num = int(fname[:config.participantNumLen])
    except:
        return False

    try:
        remainder = fname.replace(str(subject_num)+"_", "")
    except:
        return False

    remainder = remainder.replace("_nofilter", "")

    if remainder == task:
        return True
    else:
        return False


# takes one positional argument, path of study folder
class StudyFolder:
    """
    The StudyFolder object can be used for initializing a dataset newly
    exported from the WAVi Desktop software, or similar.

    Designed to make organization, cleaning, and preprocessing of mass
    datasets dead simple.

    Parameters:
        - path: path to study folder
    """

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

    def autoclean(self, group_num=1):
        """
        For each task defined in config.tasks, performs StudyFolder.standardize
        and StudyFolder.anon, standardizing task names / file structure and
        anonymizing subject headers, leaving original fnames in translator
        stored in StudyFolder/<task>_translator.txt
        """

        for task in config.tasks:

            for irregular in config.tasks[task]:

                self.standardize(irregular, task)

            if os.path.isdir(self.path + "/" + task):

                self.anon(task, group_num=group_num)

                self.no_filter_rename(task)

        self.set_raw_fnames

        if len(self.raw_fnames) > 0:

            print(
                "Some raw files couldn't be automatically standardized. "
                + "You should review them in /raw before "
                + "moving forward with analysis. You can repeat this script "
                + "after adding task names and they will be handled "
                + "appropriately.")

    def set_raw_fnames(self):

        self.raw_fnames = os.listdir(self.path + "/raw")

    def get_task_fnames(self, task):

        return(os.listdir(self.path + "/" + task))

    def standardize(self, old, new):
        """
        Standardizes every filename possible, using alternative (unclean)
        fnames from the WAVi desktop which are written in the tasks dict
        in config.py
        """

        if not os.path.isdir(self.path + "/" + new):

            os.mkdir(self.path + "/" + new)

        for fname in [fname for fname in self.raw_fnames if old in fname]:

            newfname = fname.replace(old, new)

            shutil.move(
                self.path + "/raw/" + fname,
                self.path + "/" + new + "/" + newfname)

        self.set_raw_fnames()

        if len(os.listdir(self.path + "/" + new)) == 0:
            os.rmdir(self.path + "/" + new)

    def anon(self, task, group_num=1):
        """
        Anonymizes sets of standardized task data which can then be read
        into a TaskData object.
        """

        translator = {}

        subject_leads = set([
            fname.replace(task, '')[:-4]
            for fname in self.get_task_fnames(task)
            if (is_anonymous(fname, task) == False)])

        try:
            f = open(self.path + "/translator_" + task + ".txt", "r")

            highest_sub_found = int(f.readlines()[-1].split('\t')[-1][1:]) + 1

        except FileNotFoundError:
            highest_sub_found = 0

        f = open(self.path + "/translator_" + task + ".txt", "a")

        for i, lead in enumerate(subject_leads):
            ii = i + highest_sub_found

            translator[lead] = str(group_num)\
                + "0"\
                * (config.participantNumLen - len(str(ii)) - 1)\
                + str(ii)

            f.write(lead)

            f.write("\t")

            f.write(translator[lead])

            f.write("\n")

            files = [
                fname for fname in self.get_task_fnames(task)
                if lead in fname]

            for file in files:

                newfile = file.replace(lead, translator[lead] + "_")

                shutil.move(
                    self.path + "/" + task + "/" + file,
                    self.path + "/" + task + "/" + newfile)

        f.close()

    def no_filter_rename(self, task):
        """
        Since alt files can be generated with different bandpass filters,
        this function exists to rename original files with '_nofilter'
        appended.
        """

        for fname in self.get_task_fnames(task):
            if fname[-4:] not in [".art", ".evt"] and\
            ("nofilter" not in fname):
                shutil.move(
                    self.path + "/" + task + "/" + fname,
                    self.path + "/" + task + "/"
                    + fname[:-4] + "_nofilter" + fname[-4:])

        self.task_fnames = self.get_task_fnames(task)

    def reverse_no_filter_others(self, task):
        """
        Reverses no_filter_rename on .evt and .art files
        """

        for fname in self.get_task_fnames(task):

            if fname[-4:] in [".evt", ".art"]:
                shutil.move(
                    self.path + "/" + task + "/" + fname,
                    self.path + "/" + task + "/"
                    + fname.replace("_nofilter", "")
                )
