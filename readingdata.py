import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tflayers
from pprint import pprint

filenames = ["20180128trajectory2008430.csv"]

feature_names = ['index', 'Lat', 'Long', 'Ignore', "DateP", 'Date_', 'Time_', 'dt_',  'y']
feat_defaults = [[0],     [0.],  [0.],  [0],       [0.],     ['na'],   ['na'], ['na'], ['na']  ]




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
        dataset = (tf.data.TextLineDataset(filenames)
            .skip(1)
            .map(decode_csv))
        if perform_shuffle:
            dataset = dataset.shuffle()
        dataset.batch(batch_size)
        dataset.repeat(training_epochs)
        print(dataset)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels
    return _input_fn


r_feature_columns = [
    tf.feature_column.numeric_column("lat"),
    tf.feature_column.numeric_column("lng")
]

s_feature_columns = [
    tf.feature_column.categorical_column_with_hash_bucket("date", 1000)
]


#
# index = tf.feature_column.numeric_column(key="index",  dtype='int32')
# lat = tflayers.real_valued_column("lat")
# lng = tflayers.real_valued_column("lng")
# # start here
# # ad = tflayers.real_valued_column("ad",  default_value=0, dtype='dtypes.int')
# # from tensorflow import dtypes
# # altitude = tflayers.real_valued_column("altitude", dtype='dtypes.int')
# time_before = tflayers.real_valued_column("time_before")
# # ad = tf.feature_column.numeric_column(key="ad", dtype='dtypes.int32')
# # altitude = tf.feature_column.numeric_column(key="altitude", dtype='dtypes.int32')
#
# real_feats = [index, lat, lng, time_before ]
# time = tflayers.sparse_column_with_hash_bucket('time', hash_bucket_size=1000)
# y = tflayers.sparse_column_with_keys('y', keys='bike,bus,train,subway,taxi,walk,car'.split(','))
# date = tflayers.sparse_column_with_hash_bucket('date', hash_bucket_size=1000)
# ad   = tflayers.sparse_column_with_hash_bucket('ad', hash_bucket_size=1000)
#
# sparse_feats = [y, time, date]

#
# classifier = tf.estimator.DNNLinearCombinedClassifier(
#     linear_feature_columns=s_feature_columns,
#     dnn_feature_columns=r_feature_columns,
#     dnn_hidden_units=[1000, 500, 4],
#     model_dir="tmp/")
#
# print("classifer train time: ------")
# classifier.train(input_fn=input_dataset(filenames, False), steps=1)
# print("acc time: ------")
# accuracy_score = classifier.evaluate(input_fn= input_dataset(filenames, False),
#                                      steps=50)["accuracy"]
# print('\n\n Accuracy: {0:f}'.format(accuracy_score))



dataset = tf.data.TextLineDataset('20180128trajectory2008430.csv').skip(1)
def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, feat_defaults)

    # Pack the result into a dictionary
    features = dict(zip(feature_names,fields))

    # Separate the label from the features
    label = features.pop('y')

    return features, label

features, label = _parse_line(dataset)

ds = dataset.map(_parse_line)
print(ds)



print(dataset.batch(32))

classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_names,
    n_classes=4,
    model_dir="tmp/iris")
#
classifier.train(input_fn=features,
               steps=100)
#
# accuracy_score = classifier.evaluate(input_fn=ds.get_next(),
#                                      steps=50)["accuracy"]
# print('\n\n Accuracy: {0:f}'.format(accuracy_score))

# print(dataset)
# features_result, labels_result = dataset.make_one_shot_iterator().get_next()
# print((features_result, labels_result))
