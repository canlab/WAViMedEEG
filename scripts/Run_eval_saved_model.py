import sys
sys.path.append('..')
from src import ML
from src import config
from src.Standard import SpectralAverage
import os
from tqdm import tqdm
import argParser
from datetime import datetime
import numpy as np


def main():

    args = argParser.main([
        'studies_folder',
        'study_names',
        'task',
        'data_type',
        'log_dirs',
        'checkpoint_dirs',
        'length',
        'channels',
        'artifact',
        'erp_degree',
        'filter_band',
        'normalize',
        'plot_spectra',
        'plot_hist',
        'plot_conf',
        'plot_3d_preds',
        # 'pred_level',
        'fallback',
        'combine',
        'limited_subjects'
    ])

    data_type = args.data_type
    studies_folder = args.studies_folder
    study_names = args.study_names
    log_dirs = args.log_dirs
    checkpoint_dirs = args.checkpoint_dirs
    combine = args.combine
    limited_subjects = args.limited_subjects
    fallback = args.fallback
    # pred_level = args.pred_level
    task = args.task
    length = args.length
    channels = args.channels
    artifact = args.artifact
    erp_degree = args.erp_degree
    filter_band = args.filter_band
    normalize = args.normalize
    plot_spectra = args.plot_spectra
    plot_hist = args.plot_hist
    plot_conf = args.plot_conf
    plot_3d_preds = args.plot_3d_preds

    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path =
    # "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
    if checkpoint_dirs is None:
        checkpoint_dirs = [
            log_dir + folder
            for folder in os.listdir(log_dir)
            if "_"+data_type in folder]
        checkpoint_dirs.sort()
    else:
        checkpoint_dirs = [log_dirs[0] + dir for dir in checkpoint_dirs]

    patient_paths = []
    for study_name in study_names:

        patient_path = studies_folder\
            + '/'\
            + study_name\
            + '/'\
            + data_type\
            + '/'\
            + task\
            + '_'\
            + str(length)\
            + '_'\
            + channels\
            + '_'\
            + str(artifact)

        if erp_degree is not None:
            patient_path += ("_" + str(erp_degree))

        if not os.path.isdir(patient_path):
            print("Configuration supplied was not found in study folder data.")
            print("Failed:", patient_path)
            raise FileNotFoundError
            sys.exit(3)

        patient_paths.append(patient_path)

    # Instantiate a 'Classifier' object
    myclf = ML.Classifier(data_type)

    # ============== Load All Studies' Data ==============
    for study_name, patient_path in zip(study_names, patient_paths):

        fnames = os.listdir(patient_path)
        # used for only loading in subset of subjects for evaluation
        if limited_subjects is not None:
            fnames = [fname for fname in fnames
                if fname[:config.participantNumLen] in limited_subjects]

        for fname in fnames:
            if "_"+filter_band in fname:
                myclf.LoadData(patient_path+"/"+fname)

        # fallback
        if fallback is True:
            # find unused subjects
            fallback_subs = {}
            # get translator path
            translator_file = open(
                studies_folder + "/" + study_name\
                + "/" + "translator_"+task+".txt",
                'r')
            for line in translator_file.readlines():
                subject_k = line.strip('\n').split('\t')[-1]
                if subject_k not in myclf.subjects:
                    fallback_subs[subject_k] = None

            print("Fallback 1 subjects:", fallback_subs.keys())

            # get path of one it looser artifact study folder
            fallback_patient_path = patient_path.replace(str(artifact), '1')
            # and its fnames that contain fallback subs
            fnames = [fname for fname in os.listdir(fallback_patient_path) if\
                fname[:config.participantNumLen] in fallback_subs]
            for fname in fnames:
                if "_"+filter_band in fname:
                    myclf.LoadData(fallback_patient_path+"/"+fname)
                    fallback_subs[fname[:config.participantNumLen]] = 1

            # # find unused subjects
            # fallback_subs = {}
            # # get translator path
            # translator_file = open(
            #     studies_folder + "/" + study_name\
            #     + "/" + "translator_"+task+".txt",
            #     'r')
            # for line in translator_file.readlines():
            #     subject_k = line.strip('\n').split('\t')[-1]
            #     if subject_k not in myclf.subjects:
            #         fallback_subs[subject_k] = None

            print("Fallback 2 subjects:", fallback_subs.keys())

            # get path of one it looser artifact study folder
            fallback_patient_path = patient_path.replace(str(artifact), '2')
            # and its fnames that contain fallback subs
            fnames = [fname for fname in os.listdir(fallback_patient_path) if\
                fallback_subs[fname[:config.participantNumLen]] is None]
            for fname in fnames:
                if "_"+filter_band in fname:
                    myclf.LoadData(fallback_patient_path+"/"+fname)
                    fallback_subs[fname[:config.participantNumLen]] = 2

        #
        # if combine is not True:
        #
        #     for checkpoint_dir in checkpoint_dirs:
        #
        #         label_names=checkpoint_dir.split('_')[7:]
        #         label_values=[]
        #         for group in label_names:
        #             for key, value in config.group_names.items():
        #                 if group == value:
        #                     label_values.append(key)
        #
        #         myclf.Prepare(
        #             tt_split=1,
        #             labels=label_values,
        #             normalize=normalize,
        #             eval=True)
        #
        #         if data_type == 'spectra':
        #             if plot_spectra is True:
        #                 specavgObj = SpectralAverage(myclf)
        #                 specavgObj.plot(
        #                     fig_fname=checkpoint_dir+"/"
        #                     + study_name
        #                     + "_true_"
        #                     + str(datetime.now().strftime("%H-%M-%S")))
        #
        #         y_preds = myclf.eval_saved_CNN(
        #             checkpoint_dir,
        #             plot_hist=plot_hist,
        #             plot_conf=plot_conf,
        #             plot_3d_preds=plot_3d_preds,
        #             fname=study_name,
        #             # pred_level=pred_level,
        #             save_results=True)
        #
        #         for i, (pred, inputObj) in enumerate(
        #             zip(np.rint(y_preds), myclf.data)):
        #
        #             inputObj.group = myclf.groups[int(np.argmax(pred))]
        #
        #         if data_type == 'spectra':
        #             if plot_spectra is True:
        #                 specavgObj = SpectralAverage(myclf)
        #                 specavgObj.plot(
        #                     fig_fname=checkpoint_dir+"/"
        #                     + study_name
        #                     + "_pred_"
        #                     + str(datetime.now().strftime("%H-%M-%S")))

    # if combine is True:
    for checkpoint_dir in checkpoint_dirs:

        label_names=checkpoint_dir.split('_')[7:]
        label_values=[]
        for group in label_names:
            for key, value in config.group_names.items():
                if group == value:
                    label_values.append(key)

        myclf.Prepare(
            tt_split=1,
            labels=label_values,
            normalize=normalize,
            eval=True)

        if data_type == 'spectra':
            if plot_spectra is True:
                specavgObj = SpectralAverage(myclf)
                specavgObj.plot(
                    fig_fname=checkpoint_dir+"/"
                    + study_name
                    + "_true_"
                    + str(datetime.now().strftime("%H-%M-%S")))

        y_preds = myclf.eval_saved_CNN(
            checkpoint_dir,
            plot_hist=plot_hist,
            plot_conf=plot_conf,
            plot_3d_preds=plot_3d_preds,
            fname=(study_name + "_"+str(artifact)) if limited_subjects is not None\
                else study_name,
            # pred_level=pred_level,
            save_results=True,
            fallback_list=None if fallback is False else fallback_subs)

            # TODO:
            # broken, can't change label here for combination runs
            # for i, (pred, inputObj) in enumerate(
            #     zip(np.rint(y_preds), myclf.data)):
            #
            #     inputObj.group = myclf.groups[int(np.argmax(pred))]

            # if data_type == 'spectra':
            #     if plot_spectra is True:
            #         specavgObj = SpectralAverage(myclf)
            #         specavgObj.plot(
            #             fig_fname=checkpoint_dir+"/"
            #             + study_name
            #             + "_pred_"
            #             + str(datetime.now().strftime("%H-%M-%S")))

if __name__ == '__main__':
    main()
