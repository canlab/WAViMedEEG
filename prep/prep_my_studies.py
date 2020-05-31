import config
import os
import importlib

studyDirs = os.listdir(config.myStudies)
studyDirs = [config.myStudies+"/"+study for study in studyDirs]

i = 0
for study in studyDirs:
    print("Working on study directory:", study)
    print("===================\n")
    config.studyDirectory = study
    if i == 0:
        from prep import generate_contigs
    else:
        importlib.reload(generate_contigs)
    i+=1
