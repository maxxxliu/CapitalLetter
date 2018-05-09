import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


letters_dataframe = pd.read_table("letter-recognition.data",sep=",")
letters_dataframe['Capital Letter'] = letters_dataframe['Capital Letter'].apply(lambda x: (ord(x)-65))
letters_dataframe = letters_dataframe.reindex(np.random.permutation(letters_dataframe.index))
letters_dataframe = letters_dataframe.head(18000)
test_set = letters_dataframe.tail(2000)
# letters_dataframe = letters_dataframe.reindex(np.random.permutation(letters_dataframe.index))
#print (letters_dataframe)

def preprocess_features(letters_dataframe):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = letters_dataframe[
    [
    #"Capital Letter",
     "x-box",
     "y-box",
     "width",
     "height",
     "onpix",
     "x-bar",
     "y-bar",
     "x2bar",
     "y2bar",
     "xybar",
     "x2ybr",
     "xy2br",
     "x-ege",
     "xegvy",
     "y-ege",
     "yegvx"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  #processed_features["rooms_per_person"] = (
  #  california_housing_dataframe["total_rooms"] /
  #  california_housing_dataframe["population"])
  return processed_features

def preprocess_targets(letters_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  #output_targets["median_house_value"] = (
  #  california_housing_dataframe["median_house_value"] / 1000.0)
  output_targets["Capital Letter"] = letters_dataframe["Capital Letter"]
  return output_targets


training_examples = preprocess_features(letters_dataframe.head(14000))
#print training_examples.describe()

training_targets = preprocess_targets(letters_dataframe.head(14000))
#print training_targets.describe()

validation_examples = preprocess_features(letters_dataframe.tail(4000))
#print validation_examples.describe()

validation_targets = preprocess_targets(letters_dataframe.tail(4000))
#print validation_targets.describe()

test_examples = preprocess_features(test_set)

test_targets = preprocess_targets(test_set)

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural net regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_classification_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets,
        test_examples,
        test_targets):
    """Trains a neural network regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A `DNNRegressor` object trained on the training data.
    """
    periods = 10
    steps_per_period = steps / periods

    # Create a DNNClassifer object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer,
        model_dir="model",
        n_classes=26

        #label_vocabulary=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["Capital Letter"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["Capital Letter"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["Capital Letter"],
                                                      num_epochs=1,
                                                      shuffle=False)
    predict_testing_input_fn = lambda: my_input_fn(test_examples,
                                                   test_targets["Capital Letter"],
                                                   num_epochs=1,
                                                   shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print "Training model..."
    print "LLE (on training data):"
    training_errors = []
    validation_errors = []
    test_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        dnn_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_classifier.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_predictions, 26)
        training_targets_one_hot = tf.keras.utils.to_categorical(training_targets)

        validation_predictions = dnn_classifier.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_predictions, 26)
        validation_targets_one_hot = tf.keras.utils.to_categorical(validation_targets)

        test_predictions = dnn_classifier.predict(input_fn=predict_testing_input_fn)
        test_predictions = np.array([item['class_ids'][0] for item in test_predictions])
        test_pred_one_hot = tf.keras.utils.to_categorical(test_predictions, 26)
        test_targets_one_hot = tf.keras.utils.to_categorical(test_targets)




        # Compute training and validation loss.
        training_log_loss_error = metrics.log_loss(training_pred_one_hot, training_targets_one_hot)
        validation_log_loss_error = metrics.log_loss(validation_pred_one_hot, validation_targets_one_hot)
        test_log_loss_error = metrics.log_loss(test_pred_one_hot, test_targets_one_hot)
        # Occasionally print the current loss.
        print "  period %02d : %0.2f" % (period, training_log_loss_error)
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss_error)
        validation_errors.append(validation_log_loss_error)
        test_errors.append(test_log_loss_error)
    print "Model training finished."

    # Output a graph of loss metrics over periods.
    plt.ylabel("LLE")
    plt.xlabel("Periods")
    plt.title("Log loss Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.plot(test_errors,label="testing")
    plt.legend()
    plt.show()

    print "Final Log loss (on training data):   %0.2f" % training_log_loss_error
    print "Final Log loss (on validation data): %0.2f" % validation_log_loss_error
    print "Final Log loss (on test data): %0.2f" % test_log_loss_error

    accuracy = metrics.accuracy_score(test_targets, test_predictions)
    print "Accuracy on test data: %0.2f" % accuracy


    return dnn_classifier

dnn_classifier = train_nn_classification_model(
    learning_rate=0.001,
    steps=1000,
    batch_size=10,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets,
    test_examples=test_examples,
    test_targets=test_targets)


#test_set = letters_dataframe.reindex(np.random.permutation(letters_dataframe.index)).head(2000)
# test_examples = preprocess_features(test_set)
# test_targets = preprocess_targets(test_set)

# predict_testing_input_fn = lambda: my_input_fn(test_examples,
#                                                test_targets["Capital Letter"],
#                                                num_epochs=1,
#                                                shuffle=False)

# test_predictions = dnn_classifier.predict(input_fn=predict_testing_input_fn)
# test_predictions = np.array([item['class_ids'][0] for item in test_predictions])

#log_loss_error = metrics.log_loss(tf.keras.utils.to_categorical(test_predictions, 26), tf.keras.utils.to_categorical(test_targets, 26))

# print test_targets
# print test_predictions
# print "Final LLE (on test data): %0.2f" % log_loss_error

# accuracy = metrics.accuracy_score(test_targets, test_predictions)
# print "Accuracy on test data: %0.2f" % accuracy


#tf.estimator.DNNClassifier.export_savedmodel(dnn_classifier, "model",)