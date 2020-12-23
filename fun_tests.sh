test -e ssshtest || wget -qhttps://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

run PEP8 pycodestyle Ex_bandpass.py Ex_cleaning.py Ex_cnn.py Ex_contigs.py Ex_spectra.py Ex_spectral_avg.py Clean.py ML.py Plots.py Prep.py Signals.py Standard.py config.py test_preprocessing.py
assert_no_stdout

# ============== Ex_cleaning.py ==============

# run group_error python3 Ex_cleaning.py --studies_folder /path/to/data --study_name mystudy --group_num 1000
# assert_exit_code 3

# run dir_error1 python3 Ex_cleaning.py --studies_folder WRONG/path/to/data --study_name mystudy --group_num 2
# assert_exit_code 3

# run dir_error2 python3 Ex_cleaning.py --studies_folder /path/to/data --study_name WRONGmystudy --group_num 3
# assert_exit_code 3


# ============== Ex_cnn.py ==============

# run data_error python3 Ex_cnn.py WRONG --studies_folder correct --study_name correct --task P300 --length 250 --channels 1111111111111111111 --artifact 0 --erp_degree correct --epochs 100 --normalize None --plot_ROC False --tt_split 0.33 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run dir_error python3 Ex_cnn.py contigs --studies_folder WRONG --study_name correct --task P300 --length 250 --channels 1111111111111111111 --artifact 0 --erp_degree correct --epochs 100 --normalize None --plot_ROC False --tt_split 0.33 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run dir_error2 python3 Ex_cnn.py contigs --studies_folder correct --study_name WRONG --task P300 --length 250 --channels 1111111111111111111 --artifact 0 --erp_degree correct --epochs 100 --normalize None --plot_ROC False --tt_split 0.33 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run task_error python3 Ex_cnn.py contigs --studies_folder correct --study_name correct --task what --length 250 --channels 1111111111111111111 --artifact 0 --erp_degree correct --epochs 100 --normalize None --plot_ROC False --tt_split 0.33 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run len_error python3 Ex_cnn.py contigs --studies_folder correct --study_name correct --task P300 --length 25000 --channels 1111111111111111111 --artifact 0 --erp_degree correct --epochs 100 --normalize None --plot_ROC False --tt_split 0.33 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run channel_error python3 Ex_cnn.py contigs --studies_folder correct --study_name correct --task P300 --length 250 --channels 1116112114111191111 --artifact 0 --erp_degree correct --epochs 100 --normalize None --plot_ROC False --tt_split 0.33 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run art_error python3 Ex_cnn.py WRONG --studies_folder correct --study_name correct --task P300 --length 250 --channels 1111111111111111111 --artifact 14 --erp_degree correct --epochs 100 --normalize None --plot_ROC False --tt_split 0.33 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run erp_deg_error python3 Ex_cnn.py WRONG --studies_folder correct --study_name correct --task P300 --length 250 --channels 1111111111111111111 --artifact 0 --erp_degree WRONG --epochs 100 --normalize None --plot_ROC False --tt_split 0.33 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run norm_error python3 Ex_cnn.py WRONG --studies_folder correct --study_name correct --task P300 --length 250 --channels 1111111111111111111 --artifact 0 --erp_degree correct --epochs 100 --normalize ohgeez --plot_ROC False --tt_split 0.33 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run tt_split_error python3 Ex_cnn.py WRONG --studies_folder correct --study_name correct --task P300 --length 250 --channels 1111111111111111111 --artifact 0 --erp_degree correct --epochs 100 --normalize None --plot_ROC False --tt_split 100 --learning_rate 0.01 --lr_decay False
# assert_exit_code 3

# run lr_error python3 Ex_cnn.py WRONG --studies_folder correct --study_name correct --task P300 --length 250 --channels 1111111111111111111 --artifact 0 --erp_degree correct --epochs 100 --normalize None --plot_ROC False --tt_split 0.33 --learning_rate 100 --lr_decay False
# assert_exit_code 3


# ============== Ex_contigs.py ==============
# run len_error python3 Ex_contigs.py 1000000 --artifact 1 --studies_folder correct --study_name correct --task P300 --spectra True --channels 1111111111111111111 --filter_band noalpha --erp False --erp_degree correct
# assert_exit_code 3

# run art_error python3 Ex_contigs.py 500 --artifact 4 --studies_folder correct --study_name correct --task P300 --spectra True --channels 1111111111111111111 --filter_band noalpha --erp False --erp_degree correct
# assert_exit_code 3

# run dir_error python3 Ex_contigs.py 500 --artifact 1 --studies_folder wrong --study_name wrong --task P300 --spectra True --channels 1111111111111111111 --filter_band noalpha --erp False --erp_degree correct
# assert_exit_code 3

# run task_error python3 Ex_contigs.py 500 --artifact 1 --studies_folder correct --study_name correct --task WRONG --spectra True --channels 1111111111111111111 --filter_band noalpha --erp False --erp_degree correct
# assert_exit_code 3

# run spectra_error python3 Ex_contigs.py 500 --artifact 1 --studies_folder correct --study_name correct --task P300 --spectra 4 --channels 1111111111111111111 --filter_band noalpha --erp False --erp_degree correct
# assert_exit_code 3

# run channel_error python3 Ex_contigs.py 500 --artifact 1 --studies_folder correct --study_name correct --task P300 --spectra True --channels WRONG --filter_band noalpha --erp False --erp_degree correct
# assert_exit_code 3

# run filter_error python3 Ex_contigs.py 500 --artifact 1 --studies_folder correct --study_name correct --task P300 --spectra True --channels 1111111111111111111 --filter_band WRONG --erp False --erp_degree correct
# assert_exit_code 3

# run erp_error python3 Ex_contigs.py 500 --artifact 1 --studies_folder correct --study_name correct --task P300 --spectra True --channels 1111111111111111111 --filter_band noalpha --erp WRONG --erp_degree correct
# assert_exit_code 3

# run erp_deg_error python3 Ex_contigs.py 500 --artifact 1 --studies_folder correct --study_name correct --task P300 --spectra True --channels 1111111111111111111 --filter_band noalpha --erp False --erp_degree WRONG
# assert_exit_code 3
