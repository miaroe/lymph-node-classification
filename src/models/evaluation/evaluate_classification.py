import os
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
import numpy as np
import pandas as pd
import random

from src.resources.loss import get_loss
from src.resources.architectures.ml_models import get_arch
from src.visualization.learning_curve import plot_learning_curve
from src.visualization.prediction_model import plot_predictions
from src.visualization.visual_model import visual_model
from src.visualization.confusion_matrix import confusion_matrix_and_report
from src.visualization.station_distribution import station_distribution_figure_and_report
from src.visualization.compare_metrics import plot_compare_metrics
from src.resources.train_config import get_config
from src.utils.get_paths import get_test_station_paths, get_frame_paths


# -----------------------------  EVALUATING ----------------------------------
def evaluate_model(trainer, reports_path, model_path, visualize_predictions, learning_curve, conf_matrix, model_layout,
                   station_distribution, compare_metrics):
    """
    Evaluate the model from trainer by loading the best weights and running the test dataset through it.
    Create reports and visualizations based on the evaluation if specified
    :param trainer:
    :param reports_path:
    :param model_path:
    :param visualize_predictions:
    :param learning_curve:
    :param conf_matrix:
    :param model_layout:
    :param station_distribution:
    :param compare_metrics:
    :return:
    """
    print("Evaluating model: " + model_path)
    steps = None # set to None to evaluate entire dataset
    config_path = os.path.join(model_path, 'config.json')

    config = get_config(config_path)
    train_config = config["train_config"]

    model = get_arch(train_config.get('model_arch'), train_config.get('instance_size'),
                     train_config.get('num_stations'), stateful=False)

    model.compile(loss=get_loss(train_config.get('loss')), optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])

    model.load_weights(filepath=os.path.join(model_path, 'best_model')).expect_partial()

    if trainer.model_type == 'sequence':
        if trainer.model_type == 'sequence' and trainer.use_gen:
            # create finite datasets for sequence model
            train_ds = trainer.train_ds.take(trainer.steps_per_epoch) # * trainer.batch_size
            val_ds = trainer.val_ds.take(trainer.validation_steps) # * trainer.batch_size
        else:
            train_ds = trainer.train_ds
            val_ds = trainer.val_ds
        test_sequence_model(trainer, train_ds, val_ds, model, trainer.seq_length, train_config, conf_matrix, station_distribution, reports_path)

    elif trainer.model_type == 'sequence_with_segmentation':
        train_ds = trainer.train_ds
        val_ds = trainer.val_ds
        test_sequence_model(trainer, train_ds, val_ds, model, trainer.seq_length, train_config, conf_matrix, station_distribution, reports_path)

    elif trainer.model_type == 'baseline':
        train_ds = trainer.train_ds
        val_ds = trainer.val_ds
        test_ds = trainer.test_ds

        if trainer.use_quality_weights:
            train_ds = train_ds.map(lambda x, y, z: (x, y))
            val_ds = val_ds.map(lambda x, y, z: (x, y))
            test_ds = test_ds.map(lambda x, y, z: (x, y))

        test_baseline_model(trainer, model, train_config, visualize_predictions, conf_matrix,
                            station_distribution, reports_path,train_ds, val_ds, test_ds)
        score_test = model.evaluate(test_ds, return_dict=True, steps=steps)
        with open(os.path.join(reports_path, 'test_metrics.txt'), 'a') as f:
            f.write(f'\n\nTest metrics for model: {model_path}\n')
            f.write(f'{"Metric":<12}{"Value"}\n')
            for metric, value in score_test.items():
                f.write(f'{metric:<12}{value:<.4f}\n')

    else:
        raise ValueError(f'Invalid model type: {trainer.model_type}')


    score_val = model.evaluate(val_ds, return_dict=True, steps=steps)

    with open(os.path.join(reports_path, 'val_metrics.txt'), 'a') as f:
        f.write(f'\n\nValidation metrics for model: {model_path}\n')
        f.write(f'{"Metric":<12}{"Value"}\n')
        for metric, value in score_val.items():
            f.write(f'{metric:<12}{value:<.4f}\n')

    # save learning curve to src/reports/figures
    if learning_curve:
        plot_learning_curve(model_path, reports_path)

    if compare_metrics:
        # only works if this was the last model trained
        #current_dir = get_latest_date_time('/home/miaroe/workspace/lymph-node-classification/output/models/')
        #print('current_dir: ', current_dir)
        #model_paths = ['/home/miaroe/workspace/lymph-node-classification/output/models/' + model for model in
        #               [current_dir]]  # to compare multiple models, add more paths manually
        model_paths = [model_path]
        model_names = [trainer.model_arch]  # saved in filename
        plot_compare_metrics(model_paths, model_names, reports_path)

    # save model layout to src/reports/figures
    if model_layout:
        visual_model(model, reports_path)


# -----------------------------  TESTING ----------------------------------

def test_baseline_model(trainer, model, train_config, visualize_predictions, conf_matrix, station_distribution, reports_path, train_ds, val_ds, test_ds):
    if visualize_predictions:
        plot_predictions(model, test_ds, trainer.model_type, trainer.pipeline.station_names, reports_path)

    # save confusion matrix to src/reports/figures and save classification report to src/reports
    if conf_matrix:
        true_labels, predictions = get_test_pred_baseline(model, test_ds, trainer.num_stations)
        confusion_matrix_and_report(true_labels, predictions, trainer.num_stations, train_config.get('stations_config'),
                                    reports_path, 'frame_')
    if station_distribution:
        station_distribution_figure_and_report(train_ds, val_ds, train_config.get('num_stations'),
                                               train_config.get('stations_config'), reports_path, test_ds)


def test_sequence_model(trainer, train_ds, val_ds, model, seq_length, train_config, conf_matrix, station_distribution, reports_path):
    true_labels_mean, predictions_mean = get_test_pred_sequence_mean(trainer, model, seq_length, reports_path)
    true_labels, predictions = get_test_pred_sequence(trainer, model, seq_length)
    if conf_matrix:
        confusion_matrix_and_report(true_labels_mean, predictions_mean, trainer.num_stations, train_config.get('stations_config'),
                                    reports_path, 'sequence_mean_')
        confusion_matrix_and_report(true_labels, predictions, trainer.num_stations,
                                    train_config.get('stations_config'),
                                    reports_path, 'sequence_')
    if station_distribution:
        station_distribution_figure_and_report(train_ds,
                                               val_ds,
                                               train_config.get('num_stations'),
                                               train_config.get('stations_config'), reports_path)


# get true labels and predictions for baseline model from the tf.data.dataset test_ds
def get_test_pred_baseline(model, test_ds, num_stations):
    # Initialize the true labels and predicted labels arrays
    true_labels = []
    pred_labels = []

    for batch in test_ds:
        images, labels = batch  # shape=(32, 256, 256, 3)
        pred_probs = model.predict(images)

        # multiclass: pred_probs has shape=(32, 9)
        if num_stations > 2:
            batch_pred_labels = np.argmax(pred_probs,
                                          axis=1)  # find predicted label for each image in batch, #has shape=(32,)

            true_labels.extend(np.argmax(labels, axis=1))
            pred_labels.extend(batch_pred_labels)

        # binary: pred_probs has shape=(32,)
        else:
            # need to convert the probability-based predicted labels into binary format by applying a threshold
            pred = (pred_probs >= 0.5).astype(int).flatten()
            true_labels.extend(labels.numpy())
            pred_labels.extend(pred)

    return true_labels, pred_labels

def pred_sequence(trainer, sequence, model):
    # load images from sequence, if sequence is shorter than seq_length, it is padded with zeros
    sequence_images = trainer.pipeline.load_image_sequence(sequence)
    if trainer.model_arch == 'mobileNetV3Small-lstm':
        sequence_images = sequence_images * 255.0
    # reshape sequence to (1, seq_length, img_height, img_width, channels)
    sequence_shape = sequence_images.shape
    sequence_5D = tf.reshape(sequence_images, (1, sequence_shape[0], sequence_shape[1], sequence_shape[2],
                                               sequence_shape[3]))

    # predict on sequence
    pred_sequence = model.predict(sequence_5D)[0]
    return pred_sequence

def pred_sequence_with_segementation(trainer, sequence, model):
    image_sequence, mask_sequence = trainer.pipeline.load_image_sequence(sequence)
    if trainer.model_arch == 'mutli-input_mobileNetV3Small-lstm':
        image_sequence = image_sequence * 255.0
        mask_sequence = mask_sequence * 255.0

    multi_input_seq_images = {'image_input': image_sequence, 'mask_input': mask_sequence}
    multi_input_seq_images = {k: tf.reshape(v, (1, v.shape[0], v.shape[1], v.shape[2], v.shape[3])) for k, v in
                                multi_input_seq_images.items()}
    pred_sequence = model.predict(multi_input_seq_images)[0]
    return pred_sequence


# get true labels and predictions for sequence model from test dataset stored in test_ds
def get_test_pred_sequence_mean(trainer, model, seq_length, reports_path):
    station_paths_list = get_test_station_paths(os.path.join(trainer.data_path, 'test'))
    print('Number of stations in test set:', len(station_paths_list))
    print('station_paths_list:', station_paths_list)

    true_labels = []
    predictions = []
    test_pred_df = [] #used to save predictions and true labels

    for station_path in station_paths_list:
        frame_paths = get_frame_paths(station_path, trainer.model_type)
        num_frames = len(frame_paths)

        # creates a list of sequences of length=seq_length,
        # if num_frames is not divisible by seq_length the last sequence will be shorter
        sequences = [frame_paths[i: i + seq_length] for i in range(0, num_frames, seq_length)]

        pred_sequences_dict = {}

        for sequence in sequences:
            pred_seq = pred_sequence(trainer, sequence, model)
            #if trainer.model_type == 'sequence': pred_seq = pred_sequence(trainer, sequence, model) #TODO: fix this
            #else: pred_seq = pred_sequence_with_segementation(trainer, sequence, model)
            pred_seq = pred_seq.tolist()
            # dict that stores length of sequence and prediction
            pred_sequences_dict[len(sequence)] = pred_seq


        # array that will store the mean prediction for each station
        # x_bar = sum(x_i * n_i) / sum(n_i)
        mean_pred = np.zeros(trainer.num_stations)
        for station in range(trainer.num_stations):
            station_pred = 0
            num_station_frames = 0
            for key, value in pred_sequences_dict.items():
                station_pred += value[station] * key
                num_station_frames += key
            mean_pred[station] = station_pred / num_station_frames
        predictions.append(np.argmax(mean_pred))
        station_folder = station_path.split('/')[-1]
        true_labels.append(trainer.stations_config[station_folder])
        labels = list(trainer.stations_config.keys())

        test_pred_df.append({'station_path': station_path,
                             'mean_pred': mean_pred.tolist(),
                             'pred_label': labels[np.argmax(mean_pred)],
                             'true_label': station_folder})

    test_pred_df = pd.DataFrame(test_pred_df)
    os.makedirs(reports_path, exist_ok=True)
    test_pred_df.to_csv(reports_path + 'test_pred_df.csv')

    return true_labels, predictions

def get_test_pred_sequence(trainer, model, seq_length):
    station_paths_list = get_test_station_paths(os.path.join(trainer.data_path, 'test'))

    true_labels = []
    predictions = []

    for station_path in station_paths_list:
        station_folder = station_path.split('/')[-1]
        frame_paths = get_frame_paths(station_path, trainer.model_type)
        num_frames = len(frame_paths)

        # creates a list of sequences of length=seq_length,
        # if num_frames is not divisible by seq_length the last sequence will be shorter
        sequences = [frame_paths[i: i + seq_length] for i in range(0, num_frames, seq_length)]

        for sequence in sequences:
            pred_seq = pred_sequence(trainer, sequence, model)
            #if trainer.model_type == 'sequence': pred_seq = pred_sequence(trainer, sequence, model) #TODO: fix this
            #else: pred_seq = pred_sequence_with_segementation(trainer, sequence, model)
            pred_seq = pred_seq.tolist()

            predictions.append(np.argmax(pred_seq))
            true_labels.append(trainer.stations_config[station_folder])

    return true_labels, predictions

