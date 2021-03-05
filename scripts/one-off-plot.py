import Plots
import config
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt

my_folders = [
    "20210131-032749_spectra_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/",
    "20210131-032828_spectra_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/",
    "20210131-032932_spectra_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/",
    "20210131-033105_spectra_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/",
    "20210131-033308_spectra_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/",
    "20210131-033542_spectra_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/",
    "20210131-033843_spectra_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/",
    "20210131-034154_spectra_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/",
    "20210131-034504_spectra_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/",
]

my_folders = ["logs/fit/"+folder for folder in my_folders]

training_logs = []

for folder in my_folders:
    training_logs.append(np.genfromtxt(folder+"training.log", delimiter=",")[1:, [0, 1, 3]])

train_aucs = []
test_aucs = []

for log_file in training_logs:
    # auc_val = auc(log_file[:, 0], log_file[:, 1])
    auc_val = log_file[-1, 1]
    train_aucs.append(auc_val)
    # auc_val = auc(log_file[:, 0], log_file[:, 2])
    auc_val = log_file[-1, 2]
    test_aucs.append(auc_val)

Plots.plot_layer_size_covariance(sizes=range(9), values=[train_aucs, test_aucs], metric='AUC Loss')
# Plots.plot_layer_size_covariance(sizes=range(9), values=, metric='Last-Epoch Validation Loss')
