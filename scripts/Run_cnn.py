import sys
sys.path.append('..')
from src import ML
from src import config
from src.Standard import SpectralAverage
import os
from tqdm import tqdm
import argParser
from datetime import datetime


def main():

    args = argParser.main([
        "studies_folder",
        "study_names",
        "task",
        "data_type",
        "length",
        "channels",
        "artifact",
        "erp_degree",
        "filter_band",
        "epochs",
        "normalize",
        "sample_weight",
        "plot_ROC",
        "plot_conf",
        "plot_3d_preds",
        "plot_spectra",
        "tt_split",
        "learning_rate",
        "lr_decay",
        "k_folds",
        "repetitions",
        "depth",
        "regularizer",
        "regularizer_param",
        "dropout",
        "hypertune",
        "balance",
        "logistic_regression"
    ])

    data_type = args.data_type
    studies_folder = args.studies_folder
    study_names = args.study_names
    task = args.task
    length = args.length
    channels = args.channels
    artifact = args.artifact
    erp_degree = args.erp_degree
    filter_band = args.filter_band
    epochs = args.epochs
    normalize = args.normalize
    sample_weight = args.sample_weight
    plot_ROC = args.plot_ROC
    plot_conf = args.plot_conf
    plot_3d_preds = args.plot_3d_preds
    plot_spectra = args.plot_spectra
    tt_split = args.tt_split
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    k_folds = args.k_folds
    repetitions = args.repetitions
    depth = args.depth
    regularizer = args.regularizer
    regularizer_param = args.regularizer_param
    # focal_loss_gamma = args.focal_loss_gamma
    dropout = args.dropout
    hypertune = args.hypertune
    balance = args.balance
    logistic_regression = args.logistic_regression

    # Instantiate a 'Classifier' Object
    myclf = ML.Classifier(data_type)

    # ============== Load All Studies' Data ==============
    patient_paths = []
    # patient_path points to our 'condition-positive' dataset
    # ex. patient_path =
    # "/wavi/EEGstudies/CANlab/spectra/P300_250_1111111111111111111_0_1"
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

    for patient_path, study_name in zip(patient_paths, study_names):
        print("Loading Data:", study_name)
        for fname in tqdm(sorted(os.listdir(patient_path))):
            if "_"+filter_band in fname:
                myclf.LoadData(patient_path+"/"+fname)

    # ============== Balance Class Data Sizes ==============
    # pops data off from the larger class until class sizes are equal
    # found in the reference folders
    if balance is True:
        myclf.Balance()

    if k_folds == 1:
        myclf.Prepare(tt_split=tt_split, normalize=normalize)

        for i in range(repetitions):
            if hypertune is False:
                model, _, _, = myclf.CNN(
                    learning_rate=learning_rate,
                    lr_decay=lr_decay,
                    epochs=epochs,
                    plot_ROC=plot_ROC,
                    plot_conf=plot_conf,
                    plot_3d_preds=plot_3d_preds,
                    depth=depth,
                    regularizer=regularizer,
                    regularizer_param=regularizer_param,
                    # focal_loss_gamma=focal_loss_gamma,
                    sample_weight=sample_weight,
                    dropout=dropout,
                    logistic_regression=logistic_regression)

                if data_type == 'spectra':
                    if plot_spectra is True:
                        specavgObj = SpectralAverage(myclf)
                        specavgObj.plot(
                            fig_fname=myclf.checkpoint_dir
                            + "/specavg_"
                            + os.path.basename(myclf.trial_name)
                            + "_train_"
                            + str(datetime.now().strftime("%H-%M-%S")))

            else:
                import tensorflow as tf
                from kerastuner.tuners import Hyperband

                tuner = Hyperband(
                    myclf.hypertune_CNN,
                    objective='val_accuracy',
                    max_epochs=100,
                    directory='logs/fit/',
                    project_name='1second-spectra')

                tuner.search_space_summary()

                stop_early = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10)

                tuner.search(
                    myclf.train_dataset,
                    myclf.train_labels,
                    epochs=100,
                    validation_data=(myclf.test_dataset, myclf.test_labels),
                    callbacks=[stop_early])

                models = tuner.get_best_models(num_models=10)

                for i, model in enumerate(models):
                    try:
                        os.mkdir('logs/fit/'+str(i))

                        model.save('logs/fit/'+str(i)+"/my_model")

                    except:
                        print("Can't save models. :(")

                tuner.results_summary()

    if (k_folds > 1) or (k_folds == -1):
        myclf.KfoldCrossVal(
            myclf.CNN,
            normalize=normalize,
            regularizer=regularizer,
            regularizer_param=regularizer_param,
            # focal_loss_gamma=focal_loss_gamma,
            repetitions=repetitions,
            dropout=dropout,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            epochs=epochs,
            plot_ROC=plot_ROC,
            plot_conf=plot_conf,
            plot_3d_preds=plot_3d_preds,
            k=k_folds,
            plot_spec_avgs=plot_spectra,
            sample_weight=sample_weight,
            logistic_regression=logistic_regression)


if __name__ == '__main__':
    main()
