import tensorflow as tf
from tensorflow.python import layers as tflayers
# from future import absolute_import
# from future import division
# from future import print_function

import numpy as np


COLUMNS = ['Lat', 'Long', 'Altitude', 'DateP', 'y']
feature_names = ['Lat', 'Long', 'Altitude', 'DateP']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.], ['na']]

filepath = ['mini_set.csv']

def my_input_fn(file_path, perform_shuffle=False, predict=False, repeat_count=1, batch_size=32):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.0], [0.0], [0.0], [0.], ['na']])
        label = tf.convert_to_tensor(parsed_line[-1:])
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything (but last element) are the features
        d = dict(zip(feature_names, features)), label
        return d
    if predict == False:
        dataset = (tf.data.TextLineDataset(file_path)  # Read text file
                   .skip(1)  # Skip header row
                   .map(decode_csv))  # Transform each elem by decode_csv
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.repeat(repeat_count).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
    else:
        batch_features, batch_labels = tf.data.Dataset.from_tensor_slices(file_path).map(decode_csv()).batch(1).make_one_shot_iterator.get_next()

    return batch_features, batch_labels


# feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

lat      = tf.feature_column.numeric_column("Lat")
lng      = tf.feature_column.numeric_column("Long")
altitude = tf.feature_column.numeric_column("Altitude")


lat_long_buckets = list(np.linspace(-180.0, 180.0, num=360))

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

real_fc = [lat, lng, altitude, crossed_lng_embedding, crossed_lat_embedding, crossed_ll_embedding]
all_fc = [lat, lng, altitude, lat_buck, lng_buck, crossed_lng_embedding, crossed_lat_embedding, crossed_ll_embedding]


my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 5,       # Retain the 10 most recent checkpoints.
)

classifier = tf.estimator.DNNLinearCombinedClassifier(
    linear_feature_columns=all_fc,
    dnn_feature_columns=real_fc,
    dnn_hidden_units = [50,20,4],
    n_classes=7,
    label_vocabulary='bike,bus,train,subway,taxi,walk,car'.split(','),
    model_dir="tmp/md",
    config=my_checkpointing_config

)

classifier.train(
    input_fn=lambda: my_input_fn(filepath, True, batch_size=32, repeat_count=10))

accuracy_score = classifier.evaluate(input_fn=lambda: my_input_fn(filepath, True, batch_size=1))["accuracy"]
print('\n\n Accuracy: {0:f}'.format(accuracy_score))

# predict_x = {
#     'Lat': [5.1, 5.9, 6.9],
#     'Long': [3.3, 3.0, 3.1],
#     'Altitude': [1.7, 4.2, 5.4],
# }
# labels = ['train', 'train', 'train']
#
#
# predictions_score = classifier.predict(
#     input_fn=lambda:lambda: my_input_fn(predict_x, predict=True, batch_size=1))

