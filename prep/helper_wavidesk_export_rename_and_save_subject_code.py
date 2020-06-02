import os, sys, shutil
import config

# another helpful example for renaming default filenames
# after exporting MATLAB-compatible data directly from the WAVi Desktop application
# note the following inputs below

target_folder = config.studyDirectory
string_to_replace = "Eyes_Closed_Resting"
replace_with = "rest"
subject_code = "0"


fnames = os.listdir(target_folder+'/raw')
fnames = [fname for fname in fnames if string_to_replace in fname]

translator = {}

subject_leads = set([fname.replace(string_to_replace, '')[:-4] for fname in fnames])
i = 0
for lead in subject_leads:
    translator[lead] = subject_code+"0"*(config.participantNumLen-len(str(i))-1)+str(i)
    i+=1

f = open(target_folder+"/translator.txt", "w")

for pair in translator:
    f.write(pair)
    f.write("\t")
    f.write(translator[pair])
    f.write("\n")

for fname in fnames:
    lead = fname.replace(string_to_replace, "")[:-4]
    newfname = fname.replace(string_to_replace, replace_with)
    newfname = newfname.replace(lead, "")
    newfname = translator[lead]+"_"+newfname
    shutil.move(target_folder+"/raw/"+fname, target_folder+"/raw/"+newfname)
