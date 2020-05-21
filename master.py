#!/usr/bin/env python3

import os
import config

from pathlib import Path

print("""\
                                  ,,,,
     ,,,,,,        ,,,,,       ,,,,,,,,,,   ,,,,,,           ,,,,,  #####
      ,,,,,,      ,,,,,,,    ,,,,,, ,,,,,,   ,,,,,,        ,,,,,,    ##
        ,,,,,,    ,,,,,,    ,,,,,     ,,,,,.   ,,,,,      ,,,,,     ,,,,
         ,,,,,,   ,,,,,,   ,,,,,       ,,,,,,   ,,,,,.   ,,,,,      ,,,,,
           ,,,,,,,,,,,,,,,,,,,,         ,,,,,,,  ,,,,,,,,,,,,       ,,,,,
             ,,,,,,, ,,,,,,,,             ,,,,     ,,,,,,,,         ,,,,,

""")

def viewStudyTree(startpath, max_depth=3):
    print("Please review your study directory. \n")
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if (max_depth!=None) | level <= max_depth:
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)), len(dirs))
            subindent = ' ' * 4 * (level+1)
            #for f in files:
            #    print('{}{}'.format(subindent, f))

viewStudyTree(config.studyDirectory, max_depth=config.max_tree_depth)

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


# Step IVa
# contig to tensor, jacknife
if (config.stepFourATrigger != "no")  & (config.stepFourATrigger != "yes"):
    config.stepFourATrigger = input("Step IVa: Do you want to run the model on your source folder and save? yes or no \n")
if config.stepFourATrigger == "yes":
    import contig_to_tensor_and_save
    viewStudyTree(config.studyDirectory)


# Step IVb
# contig to tensor, jacknife
if (config.stepFourBTrigger != "no")  & (config.stepFourBTrigger != "yes"):
    config.stepFourBTrigger = input("Step IVb: Do you want to evaluate jacknife models from source on your eval folder? yes or no \n")
if config.stepFourBTrigger == "yes":
    import contig_to_tensor_jacknife
    viewStudyTree(config.studyDirectory)


# Step IVc
# evaluate model
if (config.stepFourCTrigger != "no")  & (config.stepFourCTrigger != "yes"):
    config.stepFourCTrigger = input("Step IVc: Do you want to evaluate a saved model on your eval folder? yes or no \n")
if config.stepFourCTrigger == "yes":
    import evaluate_model
    viewStudyTree(config.studyDirectory)


# Step Va
# power spectral density
if (config.stepFiveATrigger != "no") & (config.stepFiveATrigger != "yes"):
    config.stepFiveATrigger = input("Step Va: Would you like to analyze power spectral density? yes or no \n")
if config.stepFiveATrigger == "yes":
    import contigs_to_spectral

# Step Vb
# cepstrum
if (config.stepFiveBTrigger != "no") & (config.stepFiveBTrigger != "yes"):
    config.stepFiveBTrigger = input("Step Vb: Would you like to analyze cepstrum data? yes or no \n")
if config.stepFiveBTrigger == "yes":
    import contigs_to_cepstrum



# Step V supplement
# PSD plot
if (config.stepFiveSTrigger != "no") & (config.stepFiveSTrigger != "yes"):
    config.stepFiveSTrigger = input("Step V: Would you like to plot a contig's PSDs? yes or no \n")
if config.stepFiveSTrigger == "yes":
    import plot_psd


# Step VI
# roc curve
if (config.stepSixATrigger != "no") & (config.stepSixATrigger != "yes"):
    config.stepSixATrigger = input("Step VIa: Would you like to plot ROC curve?")
if config.stepSixATrigger == "yes":
    import roc_curve

# Step VIb
# plot many PDFs
if (config.stepSixBTrigger != "no") & (config.stepSixBTrigger != "yes"):
    config.stepSixBTrigger = input("Step VIb: Would you like to plot many model-evaluation PDFs?")
if config.stepSixBTrigger == "yes":
    import plot_many_evals


# Step VII
# bandpass filter
if (config.stepSevenTrigger != "no") & (config.stepSevenTrigger != "yes"):
    config.stepSevenTrigger = input("Step VII: Would you like to run bandpass filters?")
if config.stepSevenTrigger == "yes":
    import csv_to_bandpass


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

# Step IX
# support vector machine
if (config.stepNineTrigger != "no") & (config.stepNineTrigger != "yes"):
    config.stepNineTrigger = input("Step IX: Would you like to run SVM on spectral data?")
if config.stepNineTrigger == "yes":
    import spectral_svm
