#!/usr/bin/env python3
#!pip3 install tensorflow==2.0.0-alpha0

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import sys, os, re

# Make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)

filename=sys.argv[1]
n=sys.argv[2]

LABELS=[0,1]

# def get_dataset(file_path, numSubjects):
#     dataset = tf.data.experimental.make_csv_dataset(
#         file_path,
#         batch_size=15,
#         #na_value="NaN",
#         num_epochs=1,
#     )
#     return dataset
#
# raw_train_data = get_dataset(filename, n)
