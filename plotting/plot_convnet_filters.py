from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib as plt
from tqdm import tqdm
import config
import sys, os, re
import numpy as np
import tensorflow as tf

import convnet

modelvar = tf.keras.models.load_model(config.model_file)

modelvar.summary()
