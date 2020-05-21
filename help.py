import os
from pathlib import Path

def viewStudyTree(startpath, max_depth):
    print("\nYour study directory is shown below.")
    print(startpath)
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if level <= max_depth:
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)), len(dirs))
            subindent = ' ' * 4 * (level+1)
            #for f in files:
            #    print('{}{}'.format(subindent, f))
