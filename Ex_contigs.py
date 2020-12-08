"""
Adapted from Ex_contigs.ipynb
Requires installation of "prep", "mne", "tqdm", and "argparse"

Args:
--parentpath: str, Directory where study folders are stored.
--studyfolder: str, Folder to store contigs.
--taskname: str, Task to use from contigs.py.

LRK
"""

# Example Contig Generation

# First, we'll import the 'Prep' module, to use its 'Trials' class, os
# and TQDM, which is a handy pip-installable package 
# that gives us nice loading bars.

import Prep, os
import argparse
from tqdm import tqdm

# 'taskname' is 
taskname = "P300"

def main():
    """
    Instantiate a 'Trials' Object
    ------
    'Trials' takes one positional argument, the path of the task folder
    to be loaded in and processed.
    We'll instantiate this object for each of our study folders
    that contains this task, and operate on it to create contigs, and spectra.
    gen_contigs() =  takes one positional argument: the length of the contig in samples
    @ 250 Hz and has the following optional arguments: 
        network_channels: (string) 
            binary list of channels in order found in config.py, 
            helper function Prep.BinarizeChannels() can provide string
        artDegree: (int) 
            corresponds to degree of artifact applied,
            from that found in wavi-output .art files, default 2=none
        
    write_contigs() = writes contigs to files        
    gen_spectra() = takes the same positional argument, and same optional arguments
    write_spectra() = writes spectra to files
    """
    # Create a argparse args
    parser = argparse.ArgumentParser(description="Conditions for creating contigs.")

    parser.add_argument('--parentpath',
                        dest='parentpath',
                        type=str,
                        required=True,
                        help='str, Directory where study folders are stored.')

    parser.add_argument('--studyfolder',
                        dest='studyfolder',
                        type=str,
                        required=True,
                        help='str, Folder to store contigs.')

    parser.add_argument('--taskname',
                        dest='taskname',
                        type=str,
                        required=True,
                        help='str, Task to use from config.py.')

    # Save the arguments in "args"
    args = parser.parse_args()

    parentpath = args.parentpath
    studyfolder = args.studyfolder
    taskname = args.taskname

    studyfolders = [folder for folder in os.listdir(parentpath)]
    for studyfolder in studyfolders:
        myp300 = Prep.TaskData(parentpath+"/"+studyfolder+"/"+taskname)
        myp300.gen_contigs(750, artDegree=1)
        myp300.write_contigs()
        myp300.gen_spectra(750, artDegree=1)
        myp300.write_spectra()

    for studyfolder in studyfolders:
        if "CANlab" not in studyfolder:
            myp300 = Prep.TaskData(parentpath+"/"+studyfolder+"/"+taskname)
            # myp300.gen_contigs(250, artDegree=2, erp=True, erpDegree=1)
            # myp300.write_contigs()
            myp300.gen_spectra(250, artDegree=2, erp=True, erpDegree=1)
            myp300.write_spectra()

    for studyfolder in studyfolders:
        myp300 = Prep.TaskData(parentpath+"/"+studyfolder+"/"+taskname)
        myp300.gen_contigs(750, artDegree=2)
        myp300.write_contigs()
        myp300.gen_spectra(750, artDegree=2)
        myp300.write_spectra()

    for studyfolder in studyfolders:
        myp300 = Prep.TaskData(parentpath+"/"+studyfolder+"/"+taskname)
        myp300.gen_contigs(750, artDegree=1)
        myp300.write_contigs()
        myp300.gen_spectra(750, artDegree=1)
        myp300.write_spectra()

    for studyfolder in studyfolders:
        if "CANlab" in studyfolder:
            myp300 = Prep.TaskData(parentpath+"/"+studyfolder+"/"+taskname)
            myp300.gen_contigs(1250, artDegree=0)
            myp300.write_contigs()
            myp300.gen_spectra(1250, artDegree=0)
            myp300.write_spectra()

    for studyfolder in studyfolders:
        if "CANlab" in studyfolder:
            myp300 = Prep.TaskData(parentpath+"/"+studyfolder+"/"+taskname)
            myp300.gen_contigs(1250, artDegree=1)
            myp300.write_contigs()
            myp300.gen_spectra(1250, artDegree=1)
            myp300.write_spectra()

    for studyfolder in studyfolders:
        if "CANlab" in studyfolder:
            myp300 = Prep.TaskData(parentpath+"/"+studyfolder+"/"+taskname)
            myp300.gen_contigs(1250, artDegree=2)
            myp300.write_contigs()
            myp300.gen_spectra(1250, artDegree=2)
            myp300.write_spectra()

if __name__ == '__main__':
    main()
