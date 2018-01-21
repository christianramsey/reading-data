import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import numpy as np
import tensorflow.contrib.learn as tflearn
import reading_data_distributed




def wide_and_deep_model(output_dir, nbuckets=5, hidden_units='64,32', learning_rate=0.01):
    real, sparse = reading_data_distributed.get_features()

    # the lat/lon columns can be discretized to yield "air traffic corridors"
    latbuckets = np.linspace(20.0, 50.0, nbuckets).tolist()  # USA
    lonbuckets = np.linspace(-120.0, -70.0, nbuckets).tolist()  # USA
    disc = {}
    disc.update({
        'd_{}'.format(key): tflayers.bucketized_column(real[key], latbuckets) \
        for key in ['dep_lat', 'arr_lat']
    })
    disc.update({
        'd_{}'.format(key): tflayers.bucketized_column(real[key], lonbuckets) \
        for key in ['dep_lon', 'arr_lon']
    })

    # cross columns that make sense in combination
    sparse['dep_loc'] = tflayers.crossed_column([disc['d_dep_lat'], disc['d_dep_lon']], \
                                                nbuckets * nbuckets)

    # # create embeddings of all the sparse columns
    # embed = {
    #     colname: create_embed(col) \
    #     for colname, col in sparse.items()
    # }
    # real.update(embed)

    estimator = \
        tflearn.DNNLinearCombinedClassifier(model_dir=output_dir,
                                            linear_feature_columns=sparse.values(),
                                            dnn_feature_columns=real.values(),
                                            dnn_hidden_units=[10,34,3])

    return estimator

import tensorflow.contrib.metrics as tfmetrics



def my_rmse(predictions, labels, **args):
  prob_ontime = predictions[:,1]
  return tfmetrics.streaming_root_mean_squared_error(prob_ontime, labels,)

def make_experiment_fn(traindata, evaldata, num_training_epochs,
                       batch_size, nbuckets, hidden_units, learning_rate, **args):
  def _experiment_fn(output_dir):
    return tflearn.Experiment(
        get_model(output_dir, nbuckets, hidden_units, learning_rate),
        train_input_fn=read_dataset(traindata, mode=tf.contrib.learn.ModeKeys.TRAIN, num_training_epochs=num_training_epochs, batch_size=batch_size),
        eval_input_fn=read_dataset(evaldata),
        export_strategies=[saved_model_export_utils.make_export_strategy(
            serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1
        )],
        eval_metrics = {
	    'rmse' : tflearn.MetricSpec(metric_fn=my_rmse, prediction_key='probabilities'),
            'training/hptuning/metric' : tflearn.MetricSpec(metric_fn=my_rmse, prediction_key='probabilities')
        },
        min_eval_frequency = 100,
        **args
    )
  return _experiment_fn