python3 ../scripts/Run_cnn.py --study_names rehab_train rehab_test WD_31-40 --erp_degree 1 --filter_band hidelta --epochs 500 --lr_decay True --regularizer_param 0.035 --dropout 0.15 --k_folds -1

mv ../logs/fit ../logs/rehab_control_loocv

python3 ../scripts/Run_cnn.py --study_names CU_pain lyons_pain WD_31-40 --erp_degree 1 --filter_band hidelta --epochs 500 --lr_decay True --regularizer_param 0.035 --dropout 0.15 --k_folds -1

mv ../logs/fit ../logs/pain_control_loocv

python3 ../scripts/Run_cnn.py --study_names rehab_train rehab_test CU_pain lyons_pain --erp_degree 1 --filter_band hidelta --epochs 500 --lr_decay True --regularizer_param 0.035 --dropout 0.15 --k_folds -1

mv ../logs/fit ../logs/pain_rehab_loocv

python3 ../scripts/Run_cnn.py --study_names rehab_train rehab_test WD_31-40 --erp_degree 1 --filter_band hidelta --epochs 500 --lr_decay True --regularizer_param 0.035 --dropout 0.15 --tt_split 0

python3 ../scripts/Run_cnn.py --study_names CU_pain lyons_pain WD_31-40 --erp_degree 1 --filter_band hidelta --epochs 500 --lr_decay True --regularizer_param 0.035 --dropout 0.15 --tt_split 0

python3 ../scripts/Run_cnn.py --study_names rehab_train rehab_test CU_pain lyons_pain --erp_degree 1 --filter_band hidelta --epochs 500 --lr_decay True --regularizer_param 0.035 --dropout 0.15 --tt_split 0
