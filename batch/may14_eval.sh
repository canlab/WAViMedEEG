python3 ../scripts/Run_eval_saved_model.py --study_names WD_31-40 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names CU_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names lyons_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_1 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_3 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_4 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain

python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/WD_31-40_predictions.txt /wavi/EEGstudies/WD_31-40/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/CU_pain_predictions.txt /wavi/EEGstudies/CU_pain/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/lyons_pain_predictions.txt /wavi/EEGstudies/lyons_pain/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/rehab_1_predictions.txt /wavi/EEGstudies/rehab_1/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/rehab_3_predictions.txt /wavi/EEGstudies/rehab_3/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/rehab_4_predictions.txt /wavi/EEGstudies/rehab_4/translator_P300.txt


python3 ../scripts/Run_eval_saved_model.py --study_names WD_31-40 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names CU_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names lyons_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_1 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_3 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_4 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab

python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/WD_31-40_predictions.txt /wavi/EEGstudies/WD_31-40/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/CU_pain_predictions.txt /wavi/EEGstudies/CU_pain/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/lyons_pain_predictions.txt /wavi/EEGstudies/lyons_pain/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/rehab_1_predictions.txt /wavi/EEGstudies/rehab_1/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/rehab_3_predictions.txt /wavi/EEGstudies/rehab_3/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-071241_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/rehab_4_predictions.txt /wavi/EEGstudies/rehab_4/translator_P300.txt


python3 ../scripts/Run_eval_saved_model.py --study_names WD_31-40 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names CU_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names lyons_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_1 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_3 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_4 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain

python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/WD_31-40_predictions.txt /wavi/EEGstudies/WD_31-40/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/CU_pain_predictions.txt /wavi/EEGstudies/CU_pain/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/lyons_pain_predictions.txt /wavi/EEGstudies/lyons_pain/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/rehab_1_predictions.txt /wavi/EEGstudies/rehab_1/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/rehab_3_predictions.txt /wavi/EEGstudies/rehab_3/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-104222_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/rehab_4_predictions.txt /wavi/EEGstudies/rehab_4/translator_P300.txt


python3 ../scripts/Run_eval_saved_model.py --study_names WD_31-40 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names CU_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names lyons_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_1 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_3 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab
python3 ../scripts/Run_eval_saved_model.py --study_names rehab_4 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab

python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/WD_31-40_predictions.txt /wavi/EEGstudies/WD_31-40/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/CU_pain_predictions.txt /wavi/EEGstudies/CU_pain/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/lyons_pain_predictions.txt /wavi/EEGstudies/lyons_pain/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/rehab_1_predictions.txt /wavi/EEGstudies/rehab_1/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/rehab_3_predictions.txt /wavi/EEGstudies/rehab_3/translator_P300.txt
python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210513-171409_erps_P300_250_1111111111111111111_0000000000110000000_1_Pain_Rehab/rehab_4_predictions.txt /wavi/EEGstudies/rehab_4/translator_P300.txt


# python3 ../scripts/Run_eval_saved_model.py --study_names WD_31-40 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
# python3 ../scripts/Run_eval_saved_model.py --study_names CU_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
# python3 ../scripts/Run_eval_saved_model.py --study_names lyons_pain --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
# python3 ../scripts/Run_eval_saved_model.py --study_names rehab_1 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
# python3 ../scripts/Run_eval_saved_model.py --study_names rehab_3 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
# python3 ../scripts/Run_eval_saved_model.py --study_names rehab_4 --erp_degree 1 --filter_band hidelta --fallback True --log_dirs /wavi/WAViMedEEG/logs/fit/ --checkpoint_dirs 20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain
#
# python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/WD_31-40_predictions.txt /wavi/EEGstudies/WD_31-40/translator_P300.txt
# python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/CU_pain_predictions.txt /wavi/EEGstudies/CU_pain/translator_P300.txt
# python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/lyons_pain_predictions.txt /wavi/EEGstudies/lyons_pain/translator_P300.txt
# python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/rehab_1_predictions.txt /wavi/EEGstudies/rehab_1/translator_P300.txt
# python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/rehab_3_predictions.txt /wavi/EEGstudies/rehab_3/translator_P300.txt
# python3 ../scripts/pretty_evaluations.py /wavi/WAViMedEEG/logs/fit/20210512-225512_erps_P300_250_1111111111111111111_0000000000110000000_1_Control_Pain/rehab_4_predictions.txt /wavi/EEGstudies/rehab_4/translator_P300.txt
