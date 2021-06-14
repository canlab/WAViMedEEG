"""
Requires installation of "mne", "tqdm", and "argparse"
"""

# Example Contig Generation

# First, we'll import the 'Prep' module, to use its 'Trials' class, os
# and TQDM, which is a handy pip-installable package
# that gives us nice loading bars.

import sys
sys.path.append('..')
from src import Prep
from src import config
import os
import argParser
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

    args = argParser.main([
        "studies_folder",
        "study_names",
        "artifact",
        "task",
        "length",
        "filter_band",
        "erp_degree",
        "channels",
        "force",
        "use_gpu",
        "gen_spectra"
    ])

    studies_folder = args.studies_folder
    study_names = args.study_names
    artifact = args.artifact
    task = args.task
    length = args.length
    filter_band = args.filter_band
    erp_degree = args.erp_degree
    channels = args.channels
    force = args.force
    use_gpu = args.use_gpu
    gen_spectra = args.gen_spectra

    if study_names is None:

        study_names = [folder for folder in os.listdir(studies_folder)]

    else:

        study_names = study_names

    for study_name in study_names:

        if not os.path.isdir(studies_folder+"/"+study_name+"/"+task):
            continue

        myp300 = Prep.TaskData(studies_folder+"/"+study_name+"/"+task)

        print("Processing study:", study_name)

        myp300.gen_contigs(
            length,
            art_degree=artifact,
            network_channels=channels,
            filter_band=filter_band,
            erp_degree=erp_degree,
            force=force,
            use_gpu=use_gpu)

        myp300.write_contigs()

        if gen_spectra is True:
            myp300.gen_spectra(
                length,
                art_degree=artifact,
                network_channels=channels,
                filter_band=filter_band,
                erp_degree=erp_degree,
                force=force,
                use_gpu=use_gpu)

            myp300.write_spectra()


if __name__ == '__main__':
    main()
