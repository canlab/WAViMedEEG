import sys
sys.path.append('..')
from src import ML
from src import config
from src.Standard import SpectralAverage
import os
from tqdm import tqdm
import argparse
from datetime import datetime
import numpy as np
import xlsxwriter
import csv


def main():

    parser = argparse.ArgumentParser(
        description="Options for saving experiments to an .xlsx sheet ")

    # parser.add_argument('data_type',
    #                     type=str,
    #                     help="Input data type: contigs, erps, or spectra")

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

    # data_type = args.data_type
    log_dirs = args.log_dirs
    checkpoint_dirs = args.checkpoint_dirs

    lrs = [1e-05,
    1.2742749857031348e-05,
    1.623776739188721e-05,
    2.06913808111479e-05,
    2.6366508987303556e-05,
    3.359818286283781e-05,
    4.281332398719396e-05,
    5.4555947811685143e-05,
    6.951927961775606e-05,
    8.858667904100833e-05,
    0.00011288378916846884,
    0.0001438449888287663,
    0.00018329807108324357,
    0.00023357214690901214,
    0.00029763514416313193,
    0.000379269019073225,
    0.0004832930238571752,
    0.0006158482110660261,
    0.0007847599703514606,
    0.001]

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

    workbook = xlsxwriter.Workbook('Experiments.xlsx')
    for z, log_dir in enumerate(log_dirs):
        # if checkpoint_dirs == None:
        checkpoint_dirs = os.listdir(log_dir)
        checkpoint_dirs.sort()

        worksheet = workbook.add_worksheet(str(z))
        # worksheet.set_default_row(200)
        # worksheet.set_column(6, 21, 200)

        # headers
        # model config / training
        worksheet.write('A1', 'Training Studies')
        worksheet.write('B1', 'Learning Rate')
        worksheet.write('C1', 'Final Accuracy')
        worksheet.write('D1', 'Final Loss')
        worksheet.write('E1', 'Final Validation Accuracy')
        worksheet.write('F1', 'Final Validation Loss')
        # training image headers
        worksheet.write('G1', 'Training / Validation Accuracy')
        worksheet.write('H1', 'Training / Validation Loss')
        worksheet.write('I1', 'ROC Curve')
        # testing image headers
        worksheet.write('J1', 'Confusion Matrix')
        worksheet.write('K1', 'Output Nodes (3D)')
        # worksheet.write('L1', 'Eval lyons_pain (old)')
        # worksheet.write('M1', 'Eval lyons_pain (new)')
        # worksheet.write('N1', 'Eval glynn_pain')
        # worksheet.write('O1', 'Eval rehab')
        # worksheet.write('P1', 'Eval ref 24-30')
        # worksheet.write('Q1', 'Eval ref 31-40')
        # worksheet.write('R1', 'Eval ref 41-50')
        # worksheet.write('S1', 'Eval ref 51-60')
        # worksheet.write('T1', 'Eval ref 61-70')
        # worksheet.write('U1', 'Eval ref 71-80')
        # worksheet.write('V1', 'Eval ref 81+')

        # worksheet.write('J1', 'Eval CU_pain')
        # worksheet.write('K1', 'Eval CU_control')
        # worksheet.write('L1', 'Eval lyons_pain (old)')
        # worksheet.write('M1', 'Eval lyons_pain (new)')
        # worksheet.write('N1', 'Eval glynn_pain')
        # worksheet.write('O1', 'Eval rehab')
        # worksheet.write('P1', 'Eval ref 24-30')
        # worksheet.write('Q1', 'Eval ref 31-40')
        # worksheet.write('R1', 'Eval ref 41-50')
        # worksheet.write('S1', 'Eval ref 51-60')
        # worksheet.write('T1', 'Eval ref 61-70')
        # worksheet.write('U1', 'Eval ref 71-80')
        # worksheet.write('V1', 'Eval ref 81+')

        for i, checkpoint_dir in enumerate(checkpoint_dirs):
            with open(log_dir+"/"+checkpoint_dir+"/training.log", 'r') as f:
                for line in f:
                    pass
                last_line = line.strip()

            training = last_line.split(',')
            training = [val.strip() for val in training]

            # if training[0] != '19':
            #     continue
            if len(training) != 5:
                continue

            worksheet.write(i+1, 0, 'WD control vs. CU pain vs. Rehab')
            worksheet.write(i+1, 1, lrs[i//5])

            # write TF log output to the A-D cols
            worksheet.write(i+1, 2, training[1])
            worksheet.write(i+1, 3, training[2])
            worksheet.write(i+1, 4, training[3])
            worksheet.write(i+1, 5, training[4])

            # write training images
            worksheet.insert_image(i+1, 6, log_dir+"/"+checkpoint_dir+"/epoch_accuracy.png", {'y_scale': 0.35, 'x_scale': 0.7, 'object_position': 1})
            worksheet.insert_image(i+1, 7, log_dir+"/"+checkpoint_dir+"/epoch_loss.png", {'y_scale': 0.35, 'x_scale': 0.7, 'object_position': 1})
            # worksheet.insert_image(i+1, 8, "logs/"+log_dir+"/"+checkpoint_dir+"/ROC.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 8, "logs/"+log_dir+"/"+checkpoint_dir+"/ROC.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})

            # write testing images
            worksheet.insert_image(i+1, 9, log_dir+"/"+checkpoint_dir+"/confusion_matrix.png", {'y_scale': 0.5, 'x_scale': 0.7, 'object_position': 1})
            # worksheet.insert_image(i+1, 10, log_dir+"/"+checkpoint_dir+"/validation_3d_preds.gif", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 11, "logs/"+log_dir+"/"+checkpoint_dir+"/lyons_pain.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})

            # worksheet.insert_image(i+1, 9, "logs/"+log_dir+"/"+checkpoint_dir+"/CU_pain.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 10, "logs/"+log_dir+"/"+checkpoint_dir+"/CU_control.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 11, "logs/"+log_dir+"/"+checkpoint_dir+"/lyons_pain.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 12, "logs/"+log_dir+"/"+checkpoint_dir+"/lyons_pain_2.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 13, "logs/"+log_dir+"/"+checkpoint_dir+"/glynn_pain.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 14, "logs/"+log_dir+"/"+checkpoint_dir+"/rehab.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 15, "logs/"+log_dir+"/"+checkpoint_dir+"/ref 24-30.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 16, "logs/"+log_dir+"/"+checkpoint_dir+"/ref 31-40.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 17, "logs/"+log_dir+"/"+checkpoint_dir+"/ref 41-50.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 18, "logs/"+log_dir+"/"+checkpoint_dir+"/ref 51-60.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 19, "logs/"+log_dir+"/"+checkpoint_dir+"/ref 61-70.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 20, "logs/"+log_dir+"/"+checkpoint_dir+"/ref 71-80.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})
            # worksheet.insert_image(i+1, 21, "logs/"+log_dir+"/"+checkpoint_dir+"/ref 81+.png", {'y_scale': 0.75, 'x_scale': 0.75, 'object_position': 1})

            worksheet.set_row(i+1, 250)

        worksheet.set_column(6, 21, 80)


    workbook.close()



if __name__ == '__main__':
    main()
