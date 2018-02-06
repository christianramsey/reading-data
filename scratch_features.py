import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators  import  dynamic_rnn_estimator
from tensorflow.python import layers as tflayers
# from future import absolute_import
# from future import division
# from future import print_function

import numpy as np

COLUMNS =        ["Lat", "Long", "Ignore", "Altitude", "DateP", "Date_", "Time_", "dt_", "y"]
FIELD_DEFAULTS = [[0.], [0.], [0], [0.], [0.], ['na'], ['na'], ['na'], ['na']]
feature_names = COLUMNS[:-1]

# filepath = ['mini_set.csv']
filepaths = []

import glob, os

for file in glob.glob("data/outputData.csv*"):
    filepaths.append(file)

filepaths = filepaths
print(filepaths)

# filepath = tf.data.Dataset.list_files('data_w_labels/ 2018*.csv')

def my_input_fn(file_path, perform_shuffle=False, predict=False, repeat_count=10000,  batch_size=32, features = None, labels = None):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, FIELD_DEFAULTS)
        label = tf.convert_to_tensor(parsed_line[-1:])
        print(label)
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything (but last element) are the features
        d = dict(zip(feature_names, features)), label
        return d
    if predict == False:
        dataset = (tf.data.TextLineDataset(filepaths)  # Read text file
                   # .skip(1)  # Skip header row
                   .map(decode_csv))  # Transform each elem by decode_csv
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.repeat(repeat_count).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        dataset = dataset.repeat(repeat_count).batch(1)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()


    return batch_features, batch_labels


# dense feature_columns
lat      = tf.feature_column.numeric_column("Lat")
lng      = tf.feature_column.numeric_column("Long")
altitude = tf.feature_column.numeric_column("Altitude")
datep    = tf.feature_column.numeric_column("DateP")

# soarse feature_columns
date_ = tf.feature_column.categorical_column_with_hash_bucket('Date_', 3650)
time_ = tf.feature_column.categorical_column_with_hash_bucket('Time_', 1500)
dt_ = tf.feature_column.categorical_column_with_hash_bucket('dt_', 10000)


lat_long_buckets = list(np.linspace(-180.0, 180.0, num=1000))

lat_buck  = tf.feature_column.bucketized_column(
    source_column = lat,
    boundaries = lat_long_buckets )

lng_buck = tf.feature_column.bucketized_column(
    source_column = lng,
    boundaries = lat_long_buckets)



crossed_lat_lon = tf.feature_column.crossed_column(
    [lat_buck, lng_buck], 7000)


crossed_lng_embedding = tf.feature_column.embedding_column(
    categorical_column=lng_buck,
    dimension=3)
crossed_lat_embedding = tf.feature_column.embedding_column(
    categorical_column=lat_buck,
    dimension=3)
crossed_ll_embedding = tf.feature_column.embedding_column(
    categorical_column=crossed_lat_lon,
    dimension=12)


crossed_all = tf.feature_column.crossed_column(
    ['Lat', 'Long', 'Date_', 'Time_', 'dt_'], 20000)

crossed_all_embedding = tf.feature_column.embedding_column(
    categorical_column=crossed_all,
    dimension=89)

date_embedding = tf.feature_column.embedding_column(
    categorical_column=date_,
    dimension=24)

time_embedding = tf.feature_column.embedding_column(
    categorical_column=time_,
    dimension=16)

dt_embedding = tf.feature_column.embedding_column(
    categorical_column=dt_,
    dimension=224)


real_fc = [lat, lng, altitude, datep, crossed_lng_embedding, crossed_lat_embedding, crossed_ll_embedding, date_embedding, time_embedding, dt_embedding, crossed_all_embedding]
all_fc =  [lat, lng, altitude, datep, date_, time_, dt_, lat_buck, lng_buck, crossed_lng_embedding, crossed_lat_embedding, crossed_ll_embedding, date_embedding, time_embedding, dt_embedding, crossed_all, crossed_all_embedding]

real_fcr = [lat, lng, altitude]

my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 5,       # Retain the 10 most recent checkpoints.
)

class_labels = ['bike', 'bus', 'car', 'driving meet conjestion', 'moto', 'motor', 'plane', 'run', 'subway', 'taxi', 'train', 'walk']

classifier = tf.estimator.DNNLinearCombinedClassifier(
    linear_feature_columns=all_fc,
    dnn_feature_columns=real_fc,
    dnn_hidden_units = [50,20,len(class_labels)],
    n_classes=len(class_labels),
    label_vocabulary=class_labels,
    model_dir="tmp/md",
    config=my_checkpointing_config
)

classifier.train(
    input_fn=lambda: my_input_fn(filepaths, True, batch_size=220, repeat_count=100))

accuracy_score = classifier.evaluate(input_fn=lambda: my_input_fn(filepaths, False, repeat_count=1, batch_size=10000))["accuracy"]
print('\n\n Accuracy: {0:f}'.format(accuracy_score))

# import pandas as pd
# from collections import OrderedDict, defaultdict
#
#
# jetti_trips = pd.read_csv('../gps_l_data.csv')
# predict_x = {}
# predict_x = jetti_trips.to_dict(into=predict_x,  orient='list'   )
#
# print(predict_x)

# predict_x = {
#     'Lat': [5.1, 5.9, 6.9],
#     'Long': [3.3, 3.0, 3.1],
#     'Altitude': [1.7, 4.2, 5.4],
# }
#
# predictions_score = classifier.predict(
#     input_fn=lambda: my_input_fn(predict_x, features = predict_x, labels=predict_x['y'], predict=True, batch_size=1))
#
# for pred_dict, expec in zip(predictions_score, predict_x['y']):
#     template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
#
#     class_id = pred_dict['class_ids'][0]
#     probability = pred_dict['probabilities'][class_id]
#
#     print(template.format(class_labels[class_id],
#                           100 * probability, expec))