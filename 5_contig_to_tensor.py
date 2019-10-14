#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plot
import numpy as np
from numpy import genfromtxt
import csv
import sys, os, re
import pathlib

print(tf.__version__)


# In[2]:


# The CANlab/WAVi Pain Study is Pre-Formatted to resemble BIDS neuroimaging formatting
# If your study does not abide to the following structure, please revisit previous scripts
# .../StudyRepo
# -------------> /raw
# -------------------> /*.art
# -------------------> /*.eeg
# -------------------> /*.evt
# -------------> /contigs
# -------------------> /train          2:1 train:test
# -------------------> /test


# In[4]:


try:
    os.chdir("CANlabStudy/contigs")
    directory=os.getcwd()
except:
    print("I couldn't find the contigs folder.\n")
    directory = input("Please give the full path of the contigs folder: ")
    os.chdir(directory)


# In[5]:


train_path = pathlib.Path('train/')
test_path = pathlib.Path('test/')


# In[6]:


# load in image paths for training set

import random
train_image_paths = list(train_path.glob('*/*'))
train_image_paths = [str(path) for path in train_image_paths]
random.shuffle(train_image_paths)

train_count = len(train_image_paths)
print("You have", train_count, "training images.")

# load in image paths for testing set?

test_image_paths = list(test_path.glob('*/*'))
test_image_paths = [str(path) for path in test_image_paths]
random.shuffle(test_image_paths)

test_count = len(test_image_paths)
print("You have", test_count, "testing images.")


# In[7]:


#list the available labels
label_names = sorted(item.name for item in train_path.glob('*/') if item.is_dir())
print("Labels discovered:", label_names)


# In[8]:


#assign an index to each label
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print("Label indices:", label_to_index)


# In[9]:


#create a list of every file and its index label
train_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                     for path in train_image_paths]

test_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in test_image_paths]


# In[10]:


def load_numpy_stack(paths):
    numpy_stack = []
    for path in paths:
        array = genfromtxt(path, delimiter=",")
        array = array.reshape(array.shape +(1,))
        numpy_stack.append(array)
    return(numpy_stack)


# In[11]:


train_arrays = load_numpy_stack(train_image_paths)
test_arrays = load_numpy_stack(test_image_paths)


# In[12]:


train_dataset = tf.data.Dataset.from_tensor_slices((train_arrays, train_image_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_arrays, test_image_labels))


# In[13]:


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# In[50]:


def kerasModel(learn, beta1, beta2):
    model = tf.keras.Sequential([
        tf.keras.layers.Convolution2D(64, kernel_size=3, strides=3, padding="same", dilation_rate=1, activation="relu", data_format="channels_last", use_bias=False, input_shape=(1250, 19, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding="same"),
        tf.keras.layers.Flatten(input_shape=(1250, 19, 1), data_format="channels_last"),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn, beta_1=beta1, beta_2=beta2),
                 loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])
    return(model)


# In[51]:


def fitModel(model, epoch):
    model.fit(train_dataset, epochs=epoch)
    return(model)


# In[52]:


def testModel(model):
    metrics = model.evaluate(test_dataset)
    return(metrics)


# In[53]:


rates = np.arange(0.01, 0.1, 0.01)
beta1s = np.arange(0.9, 0.99, 0.01)
beta2s = np.arange(0.99, 0.999, 0.001)

# learning rate does well around 0.01 or 0.02
# beta1 does well around 0.91
# beta2 does well around 0.997

epochs = int(input("How many epochs? "))
rate = float(input("What's my learning rate? "))
beta1 = float(input("What's my beta1? "))
beta2 = float(input("What's my beta2? "))

myresults=[]

compiled = kerasModel(rate, beta1, beta2)
fitted = fitModel(compiled, epochs)
results = testModel(fitted)


# In[54]:


print("Loss: ", results[0], "\nAccuracy: ", results[1])

