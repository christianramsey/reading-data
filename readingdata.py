import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tflayers
from pprint import pprint

file_path = ["data_w_labels/*.csv"]
filenames = ["data_w_labels/20180128trajectory2008430.csv"]

feature_names = ['index', 'lat', 'lng', 'ad', "altitude", 'time_before', 'date', 'time', 'y']
feat_defaults = [[0],     [0.],  [0.],  ['0'], [0.],     [0.],          ['na'], ['na'],  [0]  ]



def input_dataset(filenames=filenames, perform_shuffle=False, mode=tf.contrib.learn.ModeKeys.TRAIN, batch_size=32, training_epochs=10):

    # load in data

    def decode_csv(line):
        parsed_line = tf.decode_csv(line, feat_defaults)
        label = parsed_line[-1:]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything (but last element) are the features
        d = dict(zip(feature_names, features)), label
        return d

    def _input_fn():
        dataset = (tf.data.TextLineDataset(np.array([0, 2.3, 3.4, 2, 32., 4343, '11/23/2018', '6:00AM', 'bus' ]))
            .skip(1)
            .map(decode_csv))
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        dataset.batch(batch_size)
        dataset.repeat(training_epochs)
        print(dataset)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels
    return _input_fn



index = tflayers.real_valued_column("index")
lat = tflayers.real_valued_column("lat")
lng = tflayers.real_valued_column("lng")
# start here
ad = tflayers.real_valued_column("ad",  default_value='0', dtype='dtypes.string')
altitude = tflayers.real_valued_column("altitude")
time_before = tflayers.real_valued_column("time_before")

real_feats = [index, lat, lng, ad, altitude, time_before ]
time = tflayers.sparse_column_with_hash_bucket('time', hash_bucket_size=1000)
y = tflayers.sparse_column_with_keys('y', keys='bike,bus,train,subway,taxi,walk,car'.split(','))
time = tflayers.sparse_column_with_hash_bucket('time', hash_bucket_size=1000)
date = tflayers.sparse_column_with_hash_bucket('date', hash_bucket_size=1000)

sparse_feats = [y, time, date]

def get_feature_cols(val):

    real = {
        colname: tflayers.real_valued_column(colname) \
        for colname in \
        ('index,lat,lng,ad,altitude,time_before').split(',')
    }
    sparse = {
        'y': tflayers.sparse_column_with_keys('y',
                                                    keys='bike,bus,train,subway,taxi,walk,car'.split(',')),

        'time': tflayers.sparse_column_with_hash_bucket('origin',
                                                          hash_bucket_size=1000),  # FIXME

        'date': tflayers.sparse_column_with_hash_bucket('dest',
                                                        hash_bucket_size=1000)  # FIXME
    }
    if (val == 'real'):
        return real
    if (val == 'sparse'):
        return sparse





classifier = tf.estimator.DNNLinearCombinedClassifier(
    linear_feature_columns=sparse_feats,
    dnn_feature_columns=real_feats,
    dnn_hidden_units=[1000, 500, 7],
    model_dir="tmp/")

print("classifer train time: ------")
classifier.train(input_fn=input_dataset(filenames, False), steps=1)
print("acc time: ------")
accuracy_score = classifier.evaluate(input_fn= input_dataset(filenames, False),
                                     steps=50)["accuracy"]
print('\n\n Accuracy: {0:f}'.format(accuracy_score))