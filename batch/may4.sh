# artifact 0
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names CU_pain --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210331-000748_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab --erp_degree 1 --filter_band hidelta
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names CU_pain --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210330-012903_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain --erp_degree 1 --filter_band hidelta
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names CU_pain --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0005_20210326-055842_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Rehab --erp_degree 1 --filter_band hidelta

python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names rehab_all_new --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210331-000748_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab --erp_degree 1 --filter_band hidelta
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names rehab_all_new --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210330-012903_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain --erp_degree 1 --filter_band hidelta
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names rehab_all_new --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0005_20210326-055842_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Rehab --erp_degree 1 --filter_band hidelta

python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names WD_31-40 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210331-000748_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab --erp_degree 1 --filter_band hidelta
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names WD_31-40 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210330-012903_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain --erp_degree 1 --filter_band hidelta
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names WD_31-40 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0005_20210326-055842_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Rehab --erp_degree 1 --filter_band hidelta

python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names lyons_pain_2 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210331-000748_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab --erp_degree 1 --filter_band hidelta
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names lyons_pain_2 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210330-012903_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain --erp_degree 1 --filter_band hidelta
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names lyons_pain_2 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0005_20210326-055842_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Rehab --erp_degree 1 --filter_band hidelta

# artifact 1
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names rehab_all_new --limited_subjects 3007 3002 3041 3030 3013 3046 3026 3021 3000 3023 3037 3050 3022 3043 3001 3028 3039 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210331-000748_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab --erp_degree 1 --filter_band hidelta --artifact 1
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names rehab_all_new --limited_subjects 3007 3002 3041 3030 3013 3046 3026 3021 3000 3023 3037 3050 3022 3043 3001 3028 3039 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210330-012903_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain --erp_degree 1 --filter_band hidelta --artifact 1
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names rehab_all_new --limited_subjects 3007 3002 3041 3030 3013 3046 3026 3021 3000 3023 3037 3050 3022 3043 3001 3028 3039 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0005_20210326-055842_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Rehab --erp_degree 1 --filter_band hidelta --artifact 1

python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names WD_31-40 --limited_subjects 1051 1026 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210331-000748_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab --erp_degree 1 --filter_band hidelta --artifact 1
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names WD_31-40 --limited_subjects 1051 1026 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210330-012903_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain --erp_degree 1 --filter_band hidelta --artifact 1
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names WD_31-40 --limited_subjects 1051 1026 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0005_20210326-055842_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Rehab --erp_degree 1 --filter_band hidelta --artifact 1

# artifact 2
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names CU_pain --limited_subjects 2006 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210331-000748_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab --erp_degree 1 --filter_band hidelta --artifact 2
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names CU_pain --limited_subjects 2006 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210330-012903_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain --erp_degree 1 --filter_band hidelta --artifact 2
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names CU_pain --limited_subjects 2006 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0005_20210326-055842_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Rehab --erp_degree 1 --filter_band hidelta --artifact 2

python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names rehab_all_new --limited_subjects 3026 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210331-000748_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab --erp_degree 1 --filter_band hidelta --artifact 2
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names rehab_all_new --limited_subjects 3026 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0002_20210330-012903_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain --erp_degree 1 --filter_band hidelta --artifact 2
python3 ../scripts/Run_eval_saved_model.py erps --pred_level subject --combine True --study_names rehab_all_new --limited_subjects 3026 --log_dir "/wavi/WAViMedEEG/logs/best_binary/" --checkpoint_dir lr-p0005_20210326-055842_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Rehab --erp_degree 1 --filter_band hidelta --artifact 2
