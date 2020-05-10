import os, sys, shutil

target = "/home/clayton/science/CANlab/EEGstudies/ref pain"

fnames = os.listdir(target+'/raw')

translator = {}

subject_leads = set([fname.replace("P300_Eyes_Closed", '')[:-4] for fname in fnames])
i = 0
for lead in subject_leads:
    translator[lead] = "2"+"0"*(2-len(str(i)))+str(i)
    i+=1

f = open(target+"/translator.txt", "w")

for pair in translator:
    f.write(pair)
    f.write("\t")
    f.write(translator[pair])
    f.write("\n")

for fname in fnames:
    lead = fname.replace("P300_Eyes_Closed", "")[:-4]
    newfname = fname.replace("P300_Eyes_Closed", "p300")
    newfname = newfname.replace(lead, "")
    newfname = translator[lead]+"_"+newfname
    shutil.move(target+"/raw/"+fname, target+"/raw/"+newfname)
