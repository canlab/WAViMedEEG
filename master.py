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
if (config.stepThreeTrigger != "no")  & (config.stepThreeTrigger != "yes"):
    config.stepThreeTrigger = input("Step III: Do you want to convert your CSV trials into contigs for CNN analysis? yes or no\n")
if config.stepThreeTrigger == "yes":
    if config.selectedTask == "":
        print("I don't know my task")
        config.selectedTask = input("Which task should I use to create contigs? \n   p300? \n   flanker? \n   chronic? \n   rest? \n")
    import csv_to_contigs
    viewStudyTree(config.studyDirectory)


# Step IV
# contig to tensor, jacknife
if (config.stepFourTrigger != "no")  & (config.stepFourTrigger != "yes"):
    config.stepFourTrigger = input("Step IV: Do you want to run the model on your contigs folder? yes or no \n")
if config.stepFourTrigger == "yes":
    import contig_to_tensor_jacknife
    viewStudyTree(config.studyDirectory)


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


# Step VII
# bandpass filter
if (config.stepSevenTrigger != "no") & (config.stepSevenTrigger != "yes"):
    config.stepSevenTrigger = input("Step VII: Would you like to run a bandpass filter?")
if config.stepSevenTrigger == "yes":
    import bandpass_filter


# Step VII Supplement
# filter plots
if (config.stepSevenSuppTrigger != "no") & (config.stepSevenSuppTrigger != "yes"):
    config.stepSevenSuppTrigger = input("Step VII Supplement: Would you like to plot contig filters?")
if config.stepSevenSuppTrigger == "yes":
    if not config.filterPlotContig:
        config.filterPlotContig = input("Enter a contig filename to plot. Ex: <101_1>")
    import plot_contig_each_filter


# Step VIII
# bandpass filter
if (config.stepEightTrigger != "no") & (config.stepEightTrigger != "yes"):
    config.stepEightTrigger = input("Step VIII: Would you like to plot some random contig?")
if config.stepEightTrigger == "yes":
    import plot_contig
