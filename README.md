# WAViMedEEG


An awesome toolbox for performing rapid machine learning analysis on your EEG data


# Table of Contents
* [About the Project](#about-the-project)
   * [Built With](#built-with)
* [Getting Started](#getting-started)
   * [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
   * [Preprocessing](#preprocessing)
      * [File Cleaning](#file-cleaning)
      * [Data Structure](#data-structure)
   * [Machine Learning](#machine-learning)
      * [Linear Discriminant Analysis](#linear-discriminant-analysis)
      * [Support Vector Machine](#support-vector-machine)
      * [Convolutional Neural Network](#convolutional-neural-network)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)
   * [Special Thanks](#special-thanks-to)


# About the Project


## Built With
* Python v3.8 and up


# Getting Started

This should help you get the WAVi toolbox set up on your machine. To get a local copy up and running, follow these simple example steps.


## Prerequisites
* Python v3.8 or higher
* The following pip-installable packages:
   * [NumPy](https://numpy.org)
   * [TensorFlow (v2.2.0 or higher)](https://tensorflow.org)
   * [Matplotlib](https://matplotlib.org)
   * [Scikitlearn](https://scikit-learn.org)
   * [MNE](https://mne.tools)
   * [TQDM](https://github.com/tqdm/tqdm)


## Installation

1. Clone the repo

`git clone https://github.com/canlab/WAViMedEEG`

2. Install dependencies

`pip3 install -r requirements.txt`

3. (Optional) Edit EEG headset, task names, and base directory setup in config.py
`vim config.py`


# Usage


## Preprocessing


### File Cleaning

Before starting, you'll need to organize your dataset into a 'study folder'. If a single dataset contains more than one class, it may be useful to store them in separate folders.

If you're exporting your data using the WAVi Desktop software, do so using the 'MATLAB/csv' export option.

Regardless of whether your data is already anonymized, we will hash it again and store the original filenames in a new text file and store it again. You should back up this translator file as soon as possible.

Your 'study folder' should be organized as follows:
```
+-- /path/on/my/computer
|   +-- /myStudy_controls
|   |   +-- /raw
|   |   |   +-- John_Doe_P300.art
|   |   |   +-- John_Doe_P300.eeg
|   |   |   +-- John_Doe_P300.evt
|   |   |   +-- Jane_Smith_P300.art
|   |   |   +-- Jane_Smith_P300.eeg
|   |   |   +-- Jane_Smith_P300.evt
|   |   |   +-- ...
|   +-- /myStudy_disease123
|   |   +-- /raw
|   |   |   +-- Jim_Disease_P300.art
|   |   |   +-- Jim_Disease_P300.eeg
|   |   |   +-- Jim_Disease_P300.evt
|   |   |   +-- Jenny_Sick_P300.art
|   |   |   +-- Jenny_Sick_P300.eeg
|   |   |   +-- Jenny_Sick_P300.evt
|   |   |   +-- ...
```

Next, we will use the [Clean](../blob/master/Clean.py) module to anonymize and standardize our data.

#####Command Line
---
[Ex_clean.py](../blob/master/Ex_cleaning.py)
`python3 Ex_clean.py --help`

<!-- | argument | type | default | description |
| --- | --- | --- | --- |
| baz | bim | bam | nuh | -->

| argument | type | default | description |
| --- | --- | --- | --- |
| studies_folder | str | config.myStudies | Path to parent folder containing study folders |
| study_name | str | config.studyDirectory | Study folder containing dataset |
| group_num | int | 1 | Group number to be assigned to dataset |


#####Writing Scripts
---
See the [cleaning](../blob/master/notebook_examples/Ex_cleaning.ipynb) jupyter notebook file for example code.

1. Instantiate a 'StudyFolder' object

`myStudy = Clean.StudyFolder('/path/on/my/computer/myStudy')`

##### class Clean.StudyFolder(path)
Parameters:
   * path: path to task folder

2. Standardize and Anonymize Study Data Automatically:

`myStudy.autoclean()`

##### method Clean.StudyFolder.autoclean()
For each task defined in config.tasks, performs StudyFolder.standardize and StudyFolder.anon, standardizing task names / file structure and anonymizing subject headers, leaving original filenames in translator stored in /myStudy/<task>_translator.txt

Note: you may need to manually update your subject numbers. The first number of a subject code indicates the subject's group number, moving forward.


### Data Structure

Next, we will use the [Prep](../blob/master/Prep.py) module to structure our data, and remove bad timepoints (such as blinks, motion, etc.).
If you'd like to create new data that has been filtered to specific frequency ranges, see [Standard](../blob/master/Standard.py) and inspect the BandFilter class.

#####Command Line
---
[Ex_bandpass.py](../blob/master/Ex_bandpass.py)
`python3 Ex_bandpass.py --help`

| argument | type | default | description |
| --- | --- | --- | --- |
| studies_folder | str | config.myStudies | Path to parent folder containing study folders |
| study_name | str | None | Study folder containing dataset. If None, performs on each folder in studies_folder |
| task | str | P300 | Task to use, from options in config.tasks |
| type | str | bandpass | Which band filter method should be applied: lowpass, highpass, bandstop, bandpass |
| band | str | delta | Frequency band used for band ranges: delta, theta, alpha, beta, gamma |


[Ex_contigs.py](../blob/master/Ex_contigs.py)
`python3 Ex_contigs.py --help`

| argument | type | default | description |
| --- | --- | --- | --- |
| length | int | (required) | Duration of input data, in number of samples @ 250 Hz (or other, specified in config.sample_rate) |
| artifact | int | 0 | Strictness of artifacting algorithm to be used: 0 - strict, 1 - some, 2 - raw |
| studies_folder | str | config.myStudies | Path to parent folder containing study folders |
| study_name | str | None | Study folder containing dataset. If None, performs on each folder in studies_folder |
| task | str | P300 | Task to use, from options in config.tasks |
| spectra | bool | True | Whether spectra should automatically be generated and written to file after making contigs |
| channels | str | 1111111111111111111 | Binary string of EEG channels to be included in analysis, in order of config.channels |
| filter_band | str | nofilter | Bandfilter to be used in analysis steps, such as 'noalpha', 'delta', or 'hialpha' |
| erp | bool | False | If True, then only contigs falling immediately after a '1' or '2' in the corresponding .evt file will be processed |
| erp_degree | int | 1 | Lowest number in .evt files which will be accepted as an erp event |


#####Writing Scripts
---
See the [contigs](../blob/master/notebook_examples/Ex_contigs.ipynb) jupyter notebook file for example code.

1. Instantiate a 'TaskData' object

`myTask = TaskData('/myStudy/task_name')`

##### Prep.TaskData(path)
Parameters:
   * path: path to task folder

#### Contigs

Contigs are generated datasets of equal-length snippets from the raw data. We can use a range of criteria to deem them 'worthy' data. Such as:
   * strict (artDegree=0): require all .art datapoints to be 0 within the timeframe
   * medium (artDegree=1): require all .art datapoints to be 0 or 1 within the timeframe
   * loose (artDegree=2): allow any datapoint to pass
   * ERP (Evoked Response Potential): use only datapoints following a stimulus, in which case a .event attribute is attached to the Prep.Contig object

1. Generate contigs (these are going to be stored in RAM temporarily as 'Contig' objects)

`myTask.gen_contigs(contigLength)`

##### method Prep.TaskData.gen_contigs(contigLength)
Generates Contig objects for every file possible in TaskData.path, appending each to TaskData.contigs

Parameters:
   * contigLength: length in samples (@ 250 Hz or config.sampleRate)
   * network_channels: default config.network_channels
   * artDegree: (int) default 0, minimum value accepted to pass as a \
      "clean" contig, when reading mask from .art file
   * ERP: (bool) default False, if True then only contigs falling immediately \
      after a "1" or a "2" in the corresponding .evt file will be accepted, \
      i.e. only evoked responses

2. (Optional) Write contigs to file

`myTask.write_contigs()`

##### method Prep.TaskData.write_contigs()
Writes TaskData.contigs objects to file, under TaskData.path / contigs or TaskData.path / erps


#### Spectra
#####Command Line
---
Spectra can be automatically created using the --spectra flag in Ex_contigs.py, will have its own script soon.

#####Writing Scripts
---
##### Prep.TaskData(path)
Parameters:
   * path: path to task folder

#### Spectra

Spectra are simply fourier-transformed contig files. They can be generated using saved contig files.

1. Generate spectra (these are going to be stored in RAM temporarily as 'Spectra' objects)

`myTask.gen_spectra(contigLength)`

##### method Prep.TaskData.gen_spectra(contigLength)
Generates Spectra objects for every file possible in TaskData.path, according to TaskData.contigs

Parameters:
   * contigLength: length in samples (@ 250 Hz or config.sampleRate)
   * network_channels: default config.network_channels
   * artDegree: (int) default 0, minimum value accepted to pass as a \
      "clean" contig, when reading mask from .art file
   * ERP: (bool) default False, if True then only contigs falling immediately \
      after a "1" or a "2" in the corresponding .evt file will be accepted, \
      i.e. only evoked responses

2. (Optional) Write spectra to file

`myTask.write_spectra()`

##### method Prep.TaskData.write_spectra()
Writes TaskData.spectra objects to file, under TaskData.path / spectra

## Machine Learning
Next, we will use the [ML](../blob/master/ML.py) module to use a myriad of machine-learning tools to create archetypes for our patient groups.
Independent of which method you select to employ, the same basic formula will load in our data.


#####Writing Scripts
---
##### ML.Classifier(type)
Class object to which we can load our data before differentiating using various ML methods.

| argument | type | default | description |
| --- | --- | --- | --- |
| type | str | (required) | Which datatype the model should expect: "spectra", "erps", "contigs" |
| network_channels | str | 1111111111111111111 | Binary string of channel names to be included in analysis |

##### method ML.Classifier.LoadData(datapath)
Loads one data at a time, appending it to the ML.Classifier.data attribute.

| argument | type | default | description |
| --- | --- | --- | --- |
| path | str | (required) | Path to file (spectra, contig, or erp) |

##### method ML.Classifier.Balance(datapath)
Knowing that reference groups are named as follows:
    - ref 24-30
    - ref 31-40
    - ref 81+
    - ...

Balances the classes of a dataset such that Classifier.data
contains an equal number of control and condition-positive
Spectra or Contig objects. New data are added with Classifier.LoadData.

| argument | type | default | description |
| --- | --- | --- | --- |
| datapath | str | (required) | Parent path of reference folders listed above |

### Linear Discriminant Analysis (LDA)

#####Command Line
---
A command-line script does not yet exist for this function.

#####Writing Scripts
---
Documentation to be added soon.

### Support Vector Machine (SVM)

#####Command Line
---
A command-line script does not yet exist for this function.

#####Writing Scripts
---
Documentation to be added soon.

### Convolutional Neural Network (CNN)

#####Command Line
---
[Ex_cnn.py](../blob/master/Ex_cnn.py)
`python3 Ex_cnn.py --help`

| argument | type | default | description |
| --- | --- | --- | --- |
| data_type | str | (required) | Input data type: contigs, erps, or spectra |
| studies_folder | str | config.myStudies | Path to parent folder containing study folders |
| study_name | str | config.studyDirectory | Study folder containing dataset. If None, performs on each folder in studies_folder |
| task | str | P300 | Task to use, from options in config.tasks |
| length | int | 250 | Duration of input data, in number of samples @ 250 Hz (or as otherwise specified in config.sample_rate) |
| channels | str | 1111111111111111111 | Binary string specifying which of the EEG channels listed in config.channel_names will be included in analysis |
| artifact | int | 0 | Strictness of artifacting algorithm to be used: 0 - strict, 1 - some, 2 - raw |
| erp_degree | int | None | Lowest number in .evt files which will be accepted as an erp event (only relevant if type == 'erps'). |
| epochs | int | 100 | Number of training iterations to be run |
| plot_ROC | bool | False | Plot sensitivity-specificity curve using validation dataset (group number == 0) |
| tt_split | float | 0.33 | Ratio of test samples to train samples |
| normalize | str | None | Which normalization technique to use: standard, minmax, None |
| learning_rate | float | 0.01 | CNN step size |
| lr_decay | bool | False | Whether learning rate should decay adhering to a 0.96 / step decay rate schedule |


#####Writing Scripts
---
See the [CNN](../blob/master/notebook_examples/Ex_cnn.ipynb) jupyter notebook file for example code.


# Roadmap

See the [open issues](https://github.com/canlab/WAViMedEEG/issues) for a list of proposed features (and known issues)


# Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are *greatly appreciated*.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request


# License

Distributed under the MIT License. [See `LICENSE`](../blob/master/LICENSE) for more information.


# Contact

Clayton Schneider - @clayton_goob - claytonjschneider@gmail.com

Project link: https://github.com/canlab/WAViMedEEG


# Acknowledgements

* [NumPy](https://numpy.org)
* [TensorFlow](https://tensorflow.org)
* [Matplotlib](https://matplotlib.org)
* [Scikit-learn](https://scikit-learn.org)
* [MNE](https://mne.tools)
* [TQDM](https://github.com/tqdm/tqdm)


## Special Thanks to:

* Lyanna Kessler
* Will Stritzel
* Francesca Arese
* Ambarish Jash
* David Joffe
* Ryan Layer
* WAVi Medical
