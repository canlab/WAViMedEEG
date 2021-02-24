import os
import sys
import shutil
from detect_delimiter import detect

def main():
    studies_folder = sys.argv[1]

    for subdir, dirs, files in os.walk(studies_folder):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath[-4:] in [".evt", ".art", ".eeg"]:
                with open(filepath) as infile, open(filepath+"Z", 'w') as outfile:
                    for line in infile:
                        outfile.write(
                            # replace any delimiter with one space
                            # and replace any multiple spaces with 1 space
                            " ".join(
                                line.strip().split()).replace(
                                    detect(
                                        line,
                                        default=","),
                                    " ").strip().replace("  ", " "))
                        outfile.write("\n")
                os.remove(filepath)
                shutil.move(filepath+"Z", filepath)

if __name__ == '__main__':
    main()
