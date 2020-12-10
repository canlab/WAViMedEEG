import os
import sys

studies_folder = sys.argv[1]

for subdir, dirs, files in os.walk(studies_folder):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath[-4:] in [".eeg"]:
            with open(filepath) as infile, open(filepath[:-4]+"Z"+filepath[-4:], 'w') as outfile:
                for line in infile:
                    outfile.write(" ".join(line.split()).replace(",", " "))
                    outfile.write("\n") # trailing space shouldn't matter

