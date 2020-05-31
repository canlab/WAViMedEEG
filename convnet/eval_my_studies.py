import config
import os
import importlib

studyDirs = os.listdir(config.myStudies)
studyDirs.remove('CANlabStudy')
studyDirs = [config.myStudies+"/"+study for study in studyDirs]

i = 0
for study in studyDirs:
    print("Working on study directory:", study)
    print("===================\n")
    config.studyDirectory = study
    if i == 0:
        from convnet import convnet_eval_subjects
    else:
        importlib.reload(convnet_eval_subjects)
    i+=1
