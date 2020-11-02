# WAViMedEEG


An awesome toolbox for performing rapid machine learning analysis on your EEG data


## Table of Contents
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


## About the Project


### Built With
* Python v3.8 and up


## Getting Started

This should help you get the WAVi toolbox set up on your machine. To get a local copy up and running, follow these simple example steps.


### Prerequisites
* Python v3.8 or higher
* The following pip-installable packages:
   * [NumPy](https://numpy.org)
   * [TensorFlow (v2.0.0-beta0 or higher)](https://tensorflow.org)
   * [Matplotlib](https://matplotlib.org)
   * [Scikitlearn](https://scikit-learn.org)
   * [MNE](https://mne.tools)
   * [TQDM](https://github.com/tqdm/tqdm)


### Installation

1. Clone the repo

`git clone https://github.com/canlab/WAViMedEEG`

2. Copy template-config.py to config.py

`cp template-config.py config.py`

3. Edit EEG headset, task names, and base directory setup in config.py
`vim config.py`


## Usage


### Preprocessing


#### File Cleaning

First, you'll want to organize your dataset into a 'study folder'.

If you're exporting your data using the WAVi Desktop software, do so using the 'MATLAB/csv' export option.

Regardless of whether your data is already anonymized, we will hash it again and store the original filenames in a new text file and store it again. You should back up this translator file as soon as possible.

```
Your 'study folder' should be organized as follows:
+-- /path/on/my/computer
|   +-- /myStudy
|   |   +-- /raw
|   |   |   +-- John_Doe_P300.art
|   |   |   +-- John_Doe_P300.eeg
|   |   |   +-- John_Doe_P300.evt
|   |   |   +-- Jane_Smith_P300.art
|   |   |   +-- Jane_Smith_P300.eeg
|   |   |   +-- Jane_Smith_P300.evt
|   |   |   +-- ...
```

Next, we will use the [Prep](../blob/master/Prep.py) module to anonymize and standardize our data.

See the [cleaning](../blob/master/Ex_cleaning.ipynb) jupyter notebook file for example code.

1. Instantiate a 'StudyFolder' object

`myStudy = Prep.StudyFolder('/path/on/my/computer/myStudy')`

###### class Prep.StudyFolder(path)
Parameters:
   * path: path to task folder

2. Standardize and Anonymize Study Data Automatically:

`myStudy.autoclean()`

###### method Prep.StudyFolder.autoclean()
For each task defined in config.tasks, performs StudyFolder.standardize and StudyFolder.anon, standardizing task names / file structure and anonymizing subject headers, leaving original filenames in translator stored in /myStudy/<task>_translator.txt

Note: you may need to manually update your subject numbers. The first number of a subject code indicates the subject's group number, moving forward.


#### Data Structure

Next, we will use the [Prep](../blob/master/Prep.py) module to structure our data, and remove bad timepoints (such as blinks, motion, etc.).

See the [contigs](../blob/master/Ex_contigs.ipynb) jupyter notebook file for example code.

1. Instantiate a 'TaskData' object

`myTask = TaskData('/myStudy/task_name')`

###### Prep.TaskData(path)
Parameters:
   * path: path to task folder

##### Contigs

Contigs are generated datasets of equal-length snippets from the raw data. We can use a range of criteria to deem them 'worthy' data. Such as:
   * strict (artDegree=0): require all .art datapoints to be 0 within the timeframe
   * medium (artDegree=1): require all .art datapoints to be 0 or 1 within the timeframe
   * loose (artDegree=2): allow any datapoint to pass
   * ERP (Evoked Response Potential): use only datapoints following a stimulus, in which case a .stim attribute is attached to the Prep.Contig object

1. Generate contigs (these are going to be stored in RAM temporarily as 'Contig' objects, or if you prefer to command-line analyze)

`myTask.gen_contigs(contigLength)`

###### Prep.TaskData.gen_contigs(contigLength)
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

###### Prep.TaskData.write_contigs()
Writes TaskData.contigs objects to file, under TaskData.path / contigs


##### Spectra


### Machine Learning


#### Linear Discriminant Analysis (LDA)


#### Support Vector Machine (SVM)


#### Convolutional Neural Network (CNN)


## Roadmap

See the [open issues](https://github.com/canlab/WAViMedEEG/issues) for a list of proposed features (and known issues)


## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are *greatly appreciated*.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request


## License

Distributed under the MIT License. [See `LICENSE`](../blob/master/LICENSE) for more information.


## Contact

Clayton Schneider - @clayton_goob - claytonjschneider@gmail.com

Project link: https://github.com/canlab/WAViMedEEG


## Acknowledgements

* [NumPy](https://numpy.org)
* [TensorFlow](https://tensorflow.org)
* [Matplotlib](https://matplotlib.org)
* [Scikit-learn](https://scikit-learn.org)
* [MNE](https://mne.tools)
* [TQDM](https://github.com/tqdm/tqdm)


### Special Thanks to:

* Lyanna Kessler
* Will Stritzel
* Francesca Arese
* Ambarish Jash
* David Joffe
* Ryan Layer
* WAVi Medical
