# python3 ../scripts/Run_cnn.py --data_type erps --study_names CU_pain WD_31-40 --erp_degree 1 --filter_band hidelta --learning_rate 0.0001 --regularizer_param 0.03 --epochs 1000
#
# python3 ../scripts/Run_cnn.py --data_type erps --study_names CU_pain rehab_4 --erp_degree 1 --filter_band hidelta --learning_rate 0.0001 --regularizer_param 0.03 --epochs 1000
#
# python3 ../scripts/Run_cnn.py --data_type erps --study_names lyons_pain WD_31-40 --erp_degree 1 --filter_band hidelta --learning_rate 0.0001 --regularizer_param 0.03 --epochs 1000
#
# python3 ../scripts/Run_cnn.py --data_type erps --study_names lyons_pain rehab_4 --erp_degree 1 --filter_band hidelta --learning_rate 0.0001 --regularizer_param 0.03 --epochs 1000

python3 ../scripts/Run_cnn.py --data_type erps --study_names rehab_4 WD_31-40 --erp_degree 1 --filter_band hidelta --learning_rate 0.0001 --regularizer_param 0.03 --epochs 1000
