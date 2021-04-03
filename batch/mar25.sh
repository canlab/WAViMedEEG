# python3 ../scripts/Run_bandpass.py --type highpass --band delta --study_name WD_41-50
# python3 ../scripts/Run_bandpass.py --type highpass --band delta --study_name WD_51-60
#
# python3 ../scripts/Run_contigs.py 250 --erp_degree 1 --filter_band hidelta --study_name WD_31-40
# python3 ../scripts/Run_contigs.py 250 --erp_degree 1 --filter_band hidelta --study_name WD_41-50
# python3 ../scripts/Run_contigs.py 250 --erp_degree 1 --filter_band hidelta --study_name WD_51-60

python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0001
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0002
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0003
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0004
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0005

python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.01 --repetitions 5 --learning_rate 0.0003
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.02 --repetitions 5 --learning_rate 0.0003
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.03 --repetitions 5 --learning_rate 0.0003
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.04 --repetitions 5 --learning_rate 0.0003
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.05 --repetitions 5 --learning_rate 0.0003

python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.3 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0003
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.35 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0003
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.4 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0003
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.45 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0003
python3 ../scripts/Run_cnn.py erps --study_names rehab_all_new WD_31-40 --erp_degree 1 --filter_band hidelta --dropout 0.5 --balance True --epochs 100 --regularizer_param 0.025 --repetitions 5 --learning_rate 0.0003
