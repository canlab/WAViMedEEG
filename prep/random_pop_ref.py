import os
import config
import numpy as np
import random
from shutil import copyfile

# this script is used to match an existing conditional
# population with an even distribution of subjects
# and fill the studyDirectory from the age-matched reference groups
# so that the classes are balanced
# note: the conditional group should NOT contain
# subjects with the indentifier "2", as they are not "controls"

targetFolder = config.studyDirectory+"/contigs/"+config.selectedTask+"_"+str(config.contigLength)
conditionSubjects = list(set([fname[:config.participantNumLen] for fname in os.listdir(targetFolder)]))
# print("Number of subjects in condition dataset:", len(conditionSubjects))
matchNum = len(conditionSubjects)
print("Number of controls needed to balance condition:", matchNum)

# list of ref contig folders (the age-matched ones)
refFolders = [config.myStudies+"/"+folder for folder in os.listdir(config.myStudies) if "ref" in folder]
refFolders = [folder+"/contigs/"+config.selectedTask+"_"+str(config.contigLength) for folder in refFolders]

refSubs = []
for folder in refFolders:
    subs = list(set([fname[:config.participantNumLen] for fname in os.listdir(folder)]))
    random.shuffle(subs)
    print(subs)
    refSubs.append(subs)

f = open(config.studyDirectory+"/match_translator.txt", 'w')
f.write("Source\t Old\t New\n")

i = 0

while i < matchNum:
    j = i % len(refFolders)
    source = refFolders[j]
    newSub = refSubs[j][0]
    refSubs[j].pop(0)
    if len(refSubs[j]) == 0:
        refFolders.pop(j)
        refSubs.pop(j)
    fnames = [fname for fname in os.listdir(source) if newSub == fname[:config.participantNumLen]]
    newSubNum = "2"+"0"*(config.participantNumLen-len(str(i))-1)+str(i)
    f.write(source + "\t" + newSub + "\t" + newSubNum + "\n")
    for fname in fnames:
        newfname = fname.replace(newSub, newSubNum, 1)
        copyfile(source+"/"+fname, targetFolder+"/"+newfname)
    i+=1
