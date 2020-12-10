"""
Requires installation of "mne", "tqdm", and "argparse"
"""

# Example Contig Generation

# First, we'll import the 'Prep' module, to use its 'Trials' class, os
# and TQDM, which is a handy pip-installable package
# that gives us nice loading bars.

import Prep
import os
import argparse
import config
import sys
from tqdm import tqdm


def main():
    """
    Instantiate a 'Trials' Object
    ------
    'Trials' takes one positional argument, the path of the task folder
    to be loaded in and processed.
    We'll instantiate this object for each of our study folders
    that contains this task, and operate on it to create contigs, and spectra.
    gen_contigs() =  takes one positional argument:
    the length of the contig in samples
    @ 250 Hz and has the following optional arguments:
        network_channels: (string)
            binary list of channels in order found in config.py,
            helper function Prep.BinarizeChannels() can provide string
        art_degree: (int)
            corresponds to degree of artifact applied,
            from that found in wavi-output .art files, default 2=none

    write_contigs() = writes contigs to files
    gen_spectra() = takes the same positional argument,
    and same optional arguments
    write_spectra() = writes spectra to files
    """
    # Create a argparse args
    parser = argparse.ArgumentParser(
        description="Conditions for creating contigs.")

    parser.add_argument('length',
                        type=int,
                        help='Duration of input data, in number of samples @ '
                        + str(config.sample_rate) + ' Hz')

    parser.add_argument('--artifact',
                        dest='artifact',
                        type=int,
                        default=0,
                        help="Strictness of artifacting algorithm to be used: "
                        + "0 - strict, 1 - some, 2 - raw")

    parser.add_argument('--studies_folder',
                        dest='studies_folder',
                        type=str,
                        default=config.myStudies,
                        help='Path to parent folder containing study folders')

    parser.add_argument('--study_name',
                        dest='study_name',
                        type=str,
                        default=None,
                        help='Study folder containing dataset')

    parser.add_argument('--task',
                        dest='task',
                        type=str,
                        default="P300",
                        help='Task to use from config.py')

    parser.add_argument('--spectra',
                        dest='spectra',
                        type=bool,
                        default=True,
                        help="Whether spectra should automatically be "
                        + "generated and written to file after making contigs")

    parser.add_argument('--channels',
                        dest='channels',
                        type=str,
                        default='1111111111111111111',
                        help="Binary string specifying which of the following "
                        + "EEG channels will be included in analysis: "
                        + str(config.channel_names))

    parser.add_argument('--filter_band',
                        dest='filter_band',
                        type=str,
                        default='nofilter',
                        help="Bandfilter to be used in analysis steps, such "
                        + "as: 'noalpha', 'delta', or 'nofilter'")

    parser.add_argument('--erp',
                        dest='erp',
                        type=bool,
                        default=False,
                        help="If True then only contigs falling immediately "
                        + "after a '1' or a '2' in the corresponding "
                        + ".evt file will be accepted, i.e. only evoked "
                        + "responses")

    parser.add_argument('--erp_degree',
                        dest='erp_degree',
                        type=int,
                        default=1,
                        help="Lowest number in .evt files which will be "
                        + "accepted as an erp event")

    # Save the arguments in "args"
    args = parser.parse_args()

    length = args.length
    artifact = args.artifact
    studies_folder = args.studies_folder
    study_name = args.study_name
    task = args.task
    spectra = args.spectra
    channels = args.channels
    filter_band = args.filter_band
    erp = args.erp
    erp_degree = args.erp_degree

    # ERROR HANDLING
    if type(length) is int is False:
        print("Length must be an integer (in Hz).")
        raise ValueError
        sys.exit(1)

    try:
        if (length <= 0) or (length > 10000):
            print("Invalid entry for length, must be between 0 and 10000.")
            raise ValueError
            sys.exit(1)
    except TypeError:
        print(
            "Invalid entry for length, "
            + "must be integer value between 0 and 10000.")
        raise ValueError
        sys.exit(1)

    if artifact not in [0, 1, 2]:
        print("Artifact must be 0, 1, 2 as an integer.")
        sys.exit(1)

    if not os.path.isdir(studies_folder):
        print(
            "Invalid entry for studies_folder, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(1)

    if not os.path.isdir(os.path.join(studies_folder, study_name)):
        print(
            "Invalid entry for study_name, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(1)

    if task not in config.tasks:
        print(
            "Invalid entry for task, "
            + "not accepted as regular task name in config.")
        raise ValueError
        sys.exit(1)

    if spectra is not bool:
        print("Spectra must be boolean, True or False.")
        raise ValueError
        sys.exit(1)

    try:
        str(channels)
    except ValueError:
        print(
            "Invalid entry for channels. Must be 19-char long string of "
            + "1s and 0s")
        raise ValueError
        sys.exit(1)

    if len(channels) != 19:
        print(
            "Invalid entry for channels. Must be 19-char long string of "
            + "1s and 0s")
        raise ValueError
        sys.exit(1)

    filterband_options = ['alpha', 'beta', 'theta', 'delta',
                          'nofilter', 'noalpha', 'nodelta', 'loalpha',
                          'nobeta', 'notheta', 'lobeta', 'lotheta',
                          'hialpha', 'hitheta', 'hidelta']

    if filter_band not in filterband_options:
        print("That is not a valid filterband option. "
              + "Please an option listed here:")
        for item in filterband_options:
            print(item)
        raise ValueError
        sys.exit(1)

    if erp is not bool:
        print("Spectra must be boolean, True or False.")
        raise ValueError
        sys.exit(1)

    if erp_degree not in [1, 2, None]:
        print("Invalid entry for erp_degree. Must be int between 1 and 2.")
        raise ValueError
        sys.exit(1)

    if study_name is None:

        study_names = [folder for folder in os.listdir(studies_folder)]

    else:

        study_names = [study_name]

    for study_name in study_names:

        myp300 = Prep.TaskData(studies_folder+"/"+study_name+"/"+task)

        print("Processing study:", study_name)

        myp300.gen_contigs(
            length,
            art_degree=artifact,
            network_channels=channels,
            filter_band=filter_band,
            erp=erp,
            erp_degree=erp_degree)

        myp300.write_contigs()

        if spectra is True:
            myp300.gen_spectra(
                length,
                art_degree=artifact,
                network_channels=channels,
                filter_band=filter_band,
                erp=erp,
                erp_degree=erp_degree)

            myp300.write_spectra()


if __name__ == '__main__':
    main()
