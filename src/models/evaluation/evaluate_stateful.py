import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
sys.path.append('/home/miaroe/workspace/lymph-node-classification')

from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall
from src.resources.config import get_stations_config
from src.resources.train_config import get_config
from src.resources.architectures.ml_models import get_arch
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from src.visualization.confusion_matrix import confusion_matrix_and_report

n_splits = 5
data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence_cv'
full_video_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/FullVideos'
class_model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-06-04/21:28:00/'
reports_path = '/home/miaroe/workspace/lymph-node-classification/reports/2024-06-04/21:28:00/'
stations_config_nr = 3
stations_config = get_stations_config(stations_config_nr)
sequence_length = 10

def get_full_video_df_paths(full_video_path, test_patients):
    video_df_paths = []
    for patient in test_patients:
        video_folder = patient.split('_')[:-2]
        video_folder = '_'.join(video_folder) + '_full_videos'
        video_path = os.path.join(full_video_path, video_folder)
        patient_folder = patient.split('_')[-2:]
        patient_folder = '_'.join(patient_folder)
        patient_video_path = os.path.join(video_path, patient_folder)
        video_df_path = os.path.join(patient_video_path, 'Sequence_001', 'labels.csv')
        video_df_paths.append(video_df_path)
    return video_df_paths

def load_stateful_model(model_path):
    config_path = os.path.join(model_path, 'config.json')
    config = get_config(config_path)
    train_config = config["train_config"]

    stateful_model = get_arch(train_config.get('model_arch'), train_config.get('instance_size'),
                     train_config.get('num_stations'), stateful=True)
    stateful_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam',
                           metrics=['accuracy', Precision(), Recall()])
    stateful_model.load_weights(os.path.join(model_path, 'best_model')).expect_partial()
    stateful_model.reset_states()
    return stateful_model

def preprocess_class_frame(frame_path):
    frame = tf.keras.utils.load_img(frame_path, color_mode='rgb', target_size=None)
    frame = np.array(frame)
    frame = frame[100:1035, 530:1658]
    frame = tf.cast(frame, tf.float32)
    frame = tf.image.resize(frame, [224, 224], method='nearest')
    #frame = frame / 127.5 - 1
    return frame

# Function to get predictions for a sequence of images
def get_predictions(sequence, model):
    sequence = np.array(sequence)  # Convert list to np.array
    sequence = np.expand_dims(sequence, axis=0)  # Model expects batch dimension
    predictions = model.predict(sequence)
    return predictions[0]


# Main loop to stream images as video
def stream_test_videos(n_splits, data_path, full_video_path, class_model_path, reports_path, stations_config, sequence_length=10):

    for fold in range(n_splits):
        stations = list(stations_config.keys())

        # get all patient names in test set for fold from csv file
        test_csv_path = os.path.join(data_path, f'fold_{fold}_v2', 'test.csv')
        test_df = pd.read_csv(test_csv_path)
        test_patients = test_df['dirname'].apply(lambda x: x.split('/')[-2]).unique()

        # get full video paths for test patients
        video_df_paths = get_full_video_df_paths(full_video_path, test_patients)

        # for statistics
        num_correct = 0
        num_total = 0
        predictions = []
        true_labels = []

        class_model_path_fold = os.path.join(class_model_path, f'fold_{fold}_v2')
        class_model = load_stateful_model(class_model_path_fold)

        for video_df_path in video_df_paths:
            sequence = []
            df = pd.read_csv(video_df_path, sep=';')
            print(video_df_path)

            # Loop through the rows of the df
            for index, row in df.iterrows():
                frame_path_from_df = row['path']
                label = row['label']

                frame = frame_path_from_df.split('\\')[-1]
                frame_path = video_df_path.replace('labels.csv', frame)

                class_frame = preprocess_class_frame(frame_path)  # Preprocess for the model
                sequence.append(class_frame)  # Add the frame to the sequence

                if len(sequence) == sequence_length:
                    # Get predictions for the current sequence
                    pred = get_predictions(sequence, class_model)
                    prediction = stations[np.argmax(pred)]
                    sequence = []  # Clear the sequence
                    #sequence.pop(0)  # remove first element in sequence

                    # for statistics check if label is in stations
                    if label in stations:
                        if prediction == label:
                            num_correct += 1
                        num_total += 1
                        predictions.append(prediction)
                        true_labels.append(label)


        print(f'Accuracy: {num_correct / num_total:.2f}')
        print('accuracy:', accuracy_score(true_labels, predictions))
        print('precision:', precision_score(true_labels, predictions, average='weighted'))
        print('recall:', recall_score(true_labels, predictions, average='weighted'))
        print(classification_report(true_labels, predictions, digits=3, target_names=stations,
                                    labels=stations))
        print('true labels:', true_labels)
        print('predictions:', predictions)

        # get number from true labels config
        true_labels = [stations_config[label] for label in true_labels]
        predictions = [stations_config[pred] for pred in predictions]
        reports_path_fold = os.path.join(reports_path, f'fold_{fold}_v2' + '/')
        confusion_matrix_and_report(true_labels, predictions, 8,
                                    stations_config, reports_path_fold, 'sequence_stateful_')

        # clear memory
        tf.keras.backend.clear_session()



if __name__ == '__main__':
    stream_test_videos(n_splits, data_path, full_video_path, class_model_path, reports_path, stations_config, sequence_length)