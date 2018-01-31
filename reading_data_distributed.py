import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tflayers
from pprint import pprint
#
file_path = ["data/dataset.csv"]
# # google cloud version
# file_path = "data/*.csv"
feature_names = ['lat', 'lng', 'ad', "altitude", 'time_before', 'date', 'time', 'y']
feat_defaults = [ [0.],  [0.],  ['na'], [0.],        [0.],         ['na'],   ['na'],  [0]  ]


def get_features_raw():
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('lat,lng,ad,altitude,time_before,y').split(',')
    }
    sparse = {
        'date': tflayers.sparse_column_with_hash_bucket('date', hash_bucket_size=1000),
        'time' : tflayers.sparse_column_with_hash_bucket('time', hash_bucket_size=1000),
        'ad': tflayers.sparse_column_with_hash_bucket('ad', hash_bucket_size=10)
    }
    return real, sparse

def get_features():
    return get_features_raw()

def input_dataset(filenames="data/dataset.csv", perform_shuffle=False, mode=tf.contrib.learn.ModeKeys.EVAL, batch_size=32, training_epochs=10):

    # load in data

    def decode_csv(line):
        parsed_line = tf.decode_csv(line, feat_defaults)
        label = parsed_line[-1:]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything (but last element) are the features
        d = dict(zip(feature_names, features)), label
        return d

    def _input_fn():
        dataset = (tf.data.TextLineDataset(file_path)
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


# tf.Session().run(input_dataset(filenames="data/dataset.csv", batch_size=32, training_epochs=5))
print(input_dataset(filenames="data/dataset.csv", batch_size=32, training_epochs=5))
#
#
# real, sparse = get_features()
#
# nn_classifier = tf.estimator.DNNLinearCombinedClassifier(
#     model_dir='model/',
#     linear_feature_columns = real.values(),
#     dnn_hidden_units=[100, 50]
# )

# feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

# print(feature_columns)


# next_batch = input_dataset(file_path, mode=tf.contrib.learn.ModeKeys.TRAIN, batch_size=32) # Will return first 32 elementsperform_shuffle=False, mode=tf.contrib.learn.ModeKeys.EVAL, batch_size=32, training_epochs=10

# with tf.Session() as sess:
#     print(next_batch)
#     first_batch = sess.run(next_batch)
# print(first_batch)

#
# nn_classifier  = tf.estimator.LinearClassifier(
#     feature_columns=feature_columns,
#     n_classes=3,
#     model_dir="tmp/iris" )
#
#
#
# train_spec = tf.estimator.TrainSpec(input_fn=input_dataset(batch_size=10,
#                                                                    training_epochs=1,
#                                                                     ))
# eval_spec = tf.estimator.EvalSpec(input_fn=input_dataset(batch_size=1,
#                                                                  perform_shuffle=False,
#                                                                  ))
#
# tf.estimator.train_and_evaluate(nn_classifier, train_spec, eval_spec)
# #
# classifier = tf.estimator.LinearClassifier(
#     n_classes=3,
#     model_dir="tmp/iris", batch_features)
#
# classifier.train(input_fn=input_dataset(batch_size=1, training_epochs=1 ),
#                steps=100)

# accuracy_score = classifier.evaluate(input_fn=input_fn(test_set),
#                                      steps=50)["accuracy"]
# print('\n\n Accuracy: {0:f}'.format(accuracy_score))



# TypeError: ({'ad': <tf.Tensor 'IteratorGetNext_1:0' shape=() dtype=float32>, 'altitude': <tf.Tensor 'IteratorGetNext_1:1' shape=() dtype=float32>, 'date': <tf.Tensor 'IteratorGetNext_1:2' shape=() dtype=string>, 'time_before': <tf.Tensor 'IteratorGetNext_1:6' shape=() dtype=float32>, 'time': <tf.Tensor 'IteratorGetNext_1:5' shape=() dtype=string>, 'lat': <tf.Tensor 'IteratorGetNext_1:3' shape=() dtype=float32>, 'lng': <tf.Tensor 'IteratorGetNext_1:4' shape=() dtype=float32>}, <tf.Tensor 'IteratorGetNext_1:7' shape=(1,) dtype=int32>) is not a Python function

# TypeError: ({'ad': <tf.Tensor 'IteratorGetNext_1:0' shape=() dtype=float32>, 'altitude': <tf.Tensor 'IteratorGetNext_1:1' shape=() dtype=float32>, 'date': <tf.Tensor 'IteratorGetNext_1:2' shape=() dtype=string>, 'time_before': <tf.Tensor 'IteratorGetNext_1:6' shape=() dtype=float32>, 'time': <tf.Tensor 'IteratorGetNext_1:5' shape=() dtype=string>, 'lat': <tf.Tensor 'IteratorGetNext_1:3' shape=() dtype=float32>, 'lng': <tf.Tensor 'IteratorGetNext_1:4' shape=() dtype=float32>}, <tf.Tensor 'IteratorGetNext_1:7' shape=(1,) dtype=int32>) is not a Python function

# ValueError: Feature (key: ad) cannot have rank 0. Give: Tensor("IteratorGetNext:0", shape=(), dtype=float32, device=/device:CPU:0)


# ValueError: Feature (key: ad) cannot have rank 0. Give: Tensor("IteratorGetNext:0", shape=(), dtype=float32, device=/device:CPU:0)
# ValueError: Feature (key: ad) cannot have rank 0. Give: Tensor("IteratorGetNext:0", shape=(), dtype=float32, device=/device:CPU:0)
