#!/usr/bin/env python3

# optional, format for scripting toolbox processes,
# stringing functions together,
# etc.
# this adds convenience for running as executable
# while still performing imports correctly

print("""\
                                  ,,,,
     ,,,,,,        ,,,,,       ,,,,,,,,,,   ,,,,,,           ,,,,,  #####
      ,,,,,,      ,,,,,,,    ,,,,,, ,,,,,,   ,,,,,,        ,,,,,,    ##
        ,,,,,,    ,,,,,,    ,,,,,     ,,,,,.   ,,,,,      ,,,,,     ,,,,
         ,,,,,,   ,,,,,,   ,,,,,       ,,,,,,   ,,,,,.   ,,,,,      ,,,,,
           ,,,,,,,,,,,,,,,,,,,,         ,,,,,,,  ,,,,,,,,,,,,       ,,,,,
             ,,,,,,, ,,,,,,,,             ,,,,     ,,,,,,,,         ,,,,,

""")

# note: WIP, currently figures are set to OPEN, not to SAVE, so still requires user intervention to proceed
# automatic figure saving is on the way

import os
import config

import viewStudyTree

# ex.
# config.studyDirectory = /path/to/myfirststudy
# from plotting import plot_pdfs_many_evals_by_subj
# config.studyDirectory = /path/to/mynextstudy
# import importlib
# importlib.reload(plot_pdfs_many_evals_by_subj)
