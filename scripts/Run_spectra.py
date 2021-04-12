"""
Requires installation of "mne", "tqdm", and "argparse"
"""

# Example Spectra Generation

# First, we'll import the 'Prep' module, to use its 'Trials' class, os
# and TQDM, which is a handy pip-installable package
# that gives us nice loading bars.
import sys
sys.path.append('..')
import src.Prep
import src.config
import os
import argparse
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
        artifact: (int)
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
                        help='(Required) Duration of input data,'
                        + ' in number of samples @ '
                        + str(config.sample_rate) + ' Hz')

    parser.add_argument('--artifact',
                        dest='artifact',
                        type=str,
                        default=''.join(map(str, config.custom_art_map)),
                        help="(Default: (custom) "
                        + ''.join(map(str, config.custom_art_map))
                        + ") Strictness of artifacting "
                        + "algorithm to be used: 0=strict, 1=some, 2=raw")

    parser.add_argument('--studies_folder',
                        dest='studies_folder',
                        type=str,
                        default=config.my_studies,
                        help="(Default: " + config.my_studies + ") Path to "
                        + "parent folder containing study folders")

    parser.add_argument('--study_name',
                        dest='study_name',
                        type=str,
                        default=None,
                        help='(Default: None) Study folder containing '
                        + 'dataset. '
                        + 'If None, performed on all studies available.')

    parser.add_argument('--task',
                        dest='task',
                        type=str,
                        default="P300",
                        help='(Default: P300) Task to use from config.py')

    parser.add_argument('--channels',
                        dest='channels',
                        type=str,
                        default='1111111111111111111',
                        help="(Default: 1111111111111111111) Binary string "
                        + "specifying which of the following "
                        + "EEG channels will be included in analysis: "
                        + str(config.channel_names))

    parser.add_argument('--filter_band',
                        dest='filter_band',
                        type=str,
                        default='nofilter',
                        help="(Default: nofilter) Bandfilter to be used in "
                        + "analysis steps, such "
                        + "as: 'noalpha', 'delta', or 'nofilter'")

    parser.add_argument('--erp_degree',
                        dest='erp_degree',
                        type=int,
                        default=None,
                        help="(Default: None) If not None, lowest number in "
                        + ".evt files which will be accepted as an erp event. "
                        + "Only contigs falling immediately after erp event, "
                        + "i.e. evoked responses, are handled.")

    parser.add_argument('--force',
                        dest='force',
                        type=bool,
                        default=False,
                        help="(Default: False) If True, will write data "
                        + "in the event that the data folder already exists. "
                        + "Warning: potential data overwrite. "
                        + "Be careful!")

    # Save the arguments in "args"
    args = parser.parse_args()

    length = args.length
    artifact = args.artifact
    studies_folder = args.studies_folder
    study_name = args.study_name
    task = args.task
    channels = args.channels
    filter_band = args.filter_band
    erp_degree = args.erp_degree
    force = args.force

    # ERROR HANDLING
    if type(length) is int is False:
        print("Length must be an integer (in Hz).")
        raise ValueError
        sys.exit(3)

    try:
        if (length <= 0) or (length > 10000):
            print("Invalid entry for length, must be between 0 and 10000.")
            raise ValueError
            sys.exit(3)
    except TypeError:
        print(
            "Invalid entry for length, "
            + "must be integer value between 0 and 10000.")
        raise ValueError
        sys.exit(3)

    try:
        if len(str(artifact)) == 19:
            for char in artifact:
                if int(char) < 0 or int(char) > 2:
                    raise ValueError

        elif artifact in ["0", "1", "2"]:
            artifact = int(artifact)

        else:
            raise ValueError

    except ValueError:
        print(
            "Invalid entry for artifact. Must be str with length 19, "
            + "or int between 0 and 2.")
        raise ValueError
        sys.exit(3)

    if not os.path.isdir(studies_folder):
        print(
            "Invalid entry for studies_folder, "
            + "path does not exist as directory.")
        raise FileNotFoundError
        sys.exit(3)

    if study_name is not None:
        if not os.path.isdir(os.path.join(studies_folder, study_name)):
            print(
                "Invalid entry for study_name, "
                + "path does not exist as directory.")
            raise FileNotFoundError
            sys.exit(3)

    if task not in config.tasks:
        print(
            "Invalid entry for task, "
            + "not accepted as regular task name in config.")
        raise ValueError
        sys.exit(3)

    try:
        str(channels)
    except ValueError:
        print(
            "Invalid entry for channels. Must be 19-char long string of "
            + "1s and 0s")
        raise ValueError
        sys.exit(3)

    if len(channels) != 19:
        print(
            "Invalid entry for channels. Must be 19-char long string of "
            + "1s and 0s")
        raise ValueError
        sys.exit(3)

    if any(band == filter_band for band in config.frequency_bands):
        pass
    elif any("no"+band == filter_band for band in config.frequency_bands):
        pass
    elif any("lo"+band == filter_band for band in config.frequency_bands):
        pass
    elif any("hi"+band == filter_band for band in config.frequency_bands):
        pass
    else:
        print("That is not a valid filterband option. "
              + "Please an option listed here:")
        for item in filterband_options:
            print(item)
        raise ValueError
        sys.exit(3)

    if erp_degree not in [1, 2, None]:
        print("Invalid entry for erp_degree. Must be None, 1, or 2.")
        raise ValueError
        sys.exit(3)

    if study_name is None:

        study_names = [folder for folder in os.listdir(studies_folder)]

    else:

        study_names = [study_name]

    for study_name in study_names:

        myp300 = Prep.TaskData(studies_folder+"/"+study_name+"/"+task)

        print("Processing study:", study_name)

        myp300.gen_spectra(
            length,
            art_degree=artifact,
            network_channels=channels,
            filter_band=filter_band,
            erp_degree=erp_degree,
            force=force)

        myp300.write_spectra()


if __name__ == '__main__':
    main()
