import ML
import sys
import os
from tqdm import tqdm
import config
import argparse
from Standard import SpectralAverage
from datetime import datetime
import numpy as np
import xlsxwriter
import csv


def main():

    parser = argparse.ArgumentParser(
        description="Options for saving experiments to an .xlsx sheet ")

    parser.add_argument('data_type',
                        type=str,
                        help="Input data type: contigs, erps, or spectra")

    parser.add_argument('--log_dirs',
                        dest='log_dirs',
                        nargs='+',
                        default=["logs/fit"],
                        help="(Default: ['logs/fit']) Parent directory for "
                        + "checkpoints.")

    parser.add_argument('--checkpoint_dirs',
                        dest='checkpoint_dirs',
                        nargs='+',
                        default=None,
                        help="(Default: None) Checkpoint directory (most "
                        + "likely found in logs/fit) containing saved model.")

    # save the variables in 'args'
    args = parser.parse_args()

    data_type = args.data_type
    log_dirs = args.log_dirs
    checkpoint_dirs = args.checkpoint_dirs

    # # ERROR HANDLING
    # if data_type not in ["erps", "spectra", "contigs"]:
    #     print(
    #         "Invalid entry for data_type. "
    #         + "Must be one of ['erps', 'contigs', 'spectra']")
    #     raise ValueError
    #     sys.exit(3)
    #
    # for log_dir in log_dirs:
    #
    # if checkpoint_dirs is not None:
    #     for checkpoint_dir in checkpoint_dirs:
    #         if not os.path.isdir(checkpoint_dir):
    #             if not os.path.isdir(log_dir+checkpoint_dir):
    #                 print(
    #                     "Invalid entry for checkpoint directory, "
    #                     + "path does not exist as directory.")
    #                 raise FileNotFoundError
    #                 sys.exit(3)
    #             else:
    #                 checkpoint_dir = log_dir+checkpoint_dir
    #
    # if checkpoint_dir is None:
    #     checkpoint_dirs = [
    #         log_dir + folder
    #         for folder in os.listdir(log_dir)
    #         if "_"+data_type in folder]
    # else:
    #     checkpoint_dirs = [checkpoint_dir]

    csvFile = open('~/Desktop/CNN Variance Testing.xlsx')
    csvfileReader = csv.reader(csvFile)

    workbook = xlsxwriter.Workbook('~/Desktop/Experiments.xlsx')
    for log_dir in log_dirs:
        checkpoint_dirs = os.listdir(log_dir)

        worksheet = workbook.add_worksheet(log_dir)
        worksheet.write('A1', 'Training Studies')
        worksheet.write('B1', 'Depth')
        worksheet.write('C1', 'Final Accuracy')
        worksheet.write('D1', 'Final Loss')
        worksheet.write('E1', 'Final Validation Accuracy')
        worksheet.write('F1', 'Final Validation Loss')
        worksheet.write('G1', 'Training / Validation Accuracy')
        worksheet.write('H1', 'Training / Validation Loss')
        worksheet.write('I1', 'ROC Curve')

        

        for i, checkpoint_dir in enumerate(checkpoint_dirs):



if __name__ == '__main__':
    main()
