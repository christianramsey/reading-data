import tensorflow as tf
# from future import absolute_import
# from future import division
# from future import print_function

import time
import logging

import numpy as np


COLUMNS        = ['Lat', 'Long', 'Altitude','DateP', 'y']
FIELD_DEFAULTS = [[0.0],[0.0],[0.0],[0.],[0.]]



filenames = ['/Users/cramsey/Documents/Distributed\ Deep\ Learning/Reading\ Data/reading-data/mini_set.csv']


# ds = tf.data.TextLineDataset(train_path).skip(1)




dataset = tf.data.Dataset.from_tensor_slices(filenames)
print(dataset)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))

print(dataset)