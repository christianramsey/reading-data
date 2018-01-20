import tensorflow as tf
import numpy as np

# TODOS
# training / testing
# load multiple sharded files across different machines
# real vs sparse values
# embed sparse_values
# load embedding

file_path = "data/dataset.csv"
# google cloud version
file_path = "data/*.csv"
feature_names = ['lat', 'lng', 'ad', "altitude", 'time_before', 'date', 'time', 'y']
feat_defaults = [ [0.],  [0.],  [0.], [0.],        [0.],         ['na'],   ['na'],  [0]  ]


def my_input_fn(file_path, batch_size = 32, perform_shuffle=False, repeat_count=1):
   def decode_csv(line):
       parsed_line = tf.decode_csv(line, feat_defaults)
       label = parsed_line[-1:] # Last element is the label
       del parsed_line[-1] # Delete last element
       features = parsed_line # Everything (but last element) are the features
       d = dict(zip(feature_names, features)), label
       return d

   dataset = (tf.data.TextLineDataset(file_path) # Read text file
       # .skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(batch_size)  # Batch size to use
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels


myds = my_input_fn(file_path)

sess = tf.Session()
from pprint import pprint
# pprint(sess.run(myds))




next_batch = my_input_fn(file_path, batch_size=1, perform_shuffle=True) # Will return 32 random elements


with tf.Session() as sess:
    first_batch = sess.run(next_batch)
pprint(first_batch)


classifier = tf.estimator.DNNClassifier(
   feature_columns=my_input_fn(), # The input features to our model
   hidden_units=[10, 10], # Two layers, each with 10 neurons
   n_classes=3,
   model_dir=PATH) # Path to where checkpoints etc are stored