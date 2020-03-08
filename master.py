#!/usr/bin/env python3

import os
import config

from pathlib import Path

def viewStudyTree(startpath):
    print("Please review your study directory. \n")
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level+1)
        #for f in files:
        #    print('{}{}'.format(subindent, f))

viewStudyTree(config.studyDirectory)

# experiment details

# Step I
# wavi to csv
if (config.stepOneTrigger != "no") & (config.stepOneTrigger != "yes"):
    config.stepOneTrigger = input("Step I: Do you want to convert your raw files to CSVs? yes or no\n")
if config.stepOneTrigger == "yes":
    import wavi_to_csv
    viewStudyTree(config.studyDirectory)

# Step II
# csv to mne
if (config.stepTwoTrigger != "no")  & (config.stepTwoTrigger != "yes"):
    config.stepTwoTrigger = input("Step II: Do you want to play in MNE? yes or no\n")
if config.stepTwoTrigger == "yes":
    import csv_to_mne
    viewStudyTree(config.studyDirectory)

# Step III
# csv to contig
# NEVER use more than one at a time! always going to clear 'contigs' folder.
if (config.stepThreeTrigger != "no")  & (config.stepThreeTrigger != "yes"):
    config.stepThreeTrigger = input("Step III: Do you want to convert your CSV trials into contigs for CNN analysis? yes or no\n")
if config.stepThreeTrigger == "yes":
    if config.selectedTask == "":
        print("I don't know my task")
        config.selectedTask = input("Which task should I use to create contigs? \n   p300? \n   flanker? \n   chronic? \n   rest? \n")
    import csv_to_contigs
    viewStudyTree(config.studyDirectory)

# Step IV
# Version A
# contig to tensor
# WIDE NETWORK
# if (config.stepFourTrigger != "no")  & (config.stepFourTrigger != "yes"):
#     config.stepFourTrigger = input("Step IV: Do you want to run the model on your contigs folder? yes or no \n")
# if config.stepFourTrigger == "yes":
#     import contig_to_tensor_orig
#     viewStudyTree(config.studyDirectory)

# Step IV
# Version B
# contig to tensor
# OMITTED PER SUBJECT NETWORK
if (config.stepFourTrigger != "no")  & (config.stepFourTrigger != "yes"):
    config.stepFourTrigger = input("Step IV: Do you want to run the model on your contigs folder? yes or no \n")
if config.stepFourTrigger == "yes":
    import contig_to_tensor_jacknife
    viewStudyTree(config.studyDirectory)

# Step IV Supplement
if (config.stepFourTriggerDistributions != "no") & (config.stepFourTriggerDistributions != "yes"):
    config.stepFourTriggerDistributions = input("Step IV: Would you like to export score distributions? yes or no \n")
if config.stepFourTriggerDistributions == "yes":
    import score_distributions_jacknife

# Step V
# frequency decomposition
if (config.stepFiveTrigger != "no") & (config.stepFiveTrigger != "yes"):
    config.stepFiveTrigger = input("Step V: Would you like to analyze power spectral density? yes or no \n")
if config.stepFiveTrigger == "yes":
    import power_spectral_density

# Step VI
# roc curve
if (config.stepSixTrigger != "no") & (config.stepSixTrigger != "yes"):
    config.stepSixTrigger = input("Step VI: Would you like to plot ROC curve?")
if config.stepSixTrigger == "yes":
    import roc_curve
