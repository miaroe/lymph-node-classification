import os

import tensorflow as tf
import time
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array

from src.resources.config import get_stations_config
from src.utils.get_paths import get_frame_paths, get_test_station_paths

model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-04-19/22:44:29'
test_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/test'
seq_length = 10
stations = list(get_stations_config(3).keys())

# Function to preprocess images
def preprocess_frame(image_path):
    frame = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(224, 224))
    frame = img_to_array(frame)
    frame = tf.cast(frame, tf.float32)
    frame = frame / 127.5 - 1
    return frame

# Function to get predictions for a sequence of images
def get_predictions(sequence, model):
    sequence = np.array(sequence)  # Convert list to np.array
    sequence = np.expand_dims(sequence, axis=0)  # Model expects batch dimension
    predictions = model.predict(sequence)
    return predictions[0]

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model
def measure_run_time(model_path, test_path, sequence_length, stations):
    test_station_paths = get_test_station_paths(test_path)
    model = load_model(os.path.join(model_path, 'best_model'))
    sequence = []
    seq_counter = 0
    seq_pred_times = []
    preprocessing_times = []
    pred_times = []

    for station_path in test_station_paths:
        frame_paths = get_frame_paths(station_path, 'sequence')
        for frame_path in frame_paths:
            start_time = time.time()
            preprocessing_start = time.time()
            frame = preprocess_frame(frame_path)
            preprocessing_end = time.time()
            preprocessing_times.append(preprocessing_end - preprocessing_start)

            sequence.append(frame)  # Add the frame to the sequence

            if len(sequence) == sequence_length:
                # Get predictions for the current sequence
                pred_time_start = time.time()
                pred = get_predictions(sequence, model)
                pred_time_end = time.time()
                prediction = stations[np.argmax(pred)]
                sequence = []
                end_time = time.time()
                print(prediction)
                seq_counter += 1
                seq_pred_times.append(end_time - start_time)
                pred_times.append(pred_time_end - pred_time_start)

    print('sequence prediction times:', seq_pred_times)
    print('preprocessing times:', preprocessing_times)
    print('prediction times:', pred_times)
    print('Number of sequences:', seq_counter)
    # only use the last 300 sequences
    print('Average sequence prediction time in ms:', np.mean(seq_pred_times[-300:])*1000)
    print('Average preprocessing time in ms:', np.mean(preprocessing_times[-300:])*1000)
    print('Average prediction time in ms:', np.mean(pred_times[-300:])*1000)

    print('Average sequence preprocessing time: {:.2f} ms ± {:.2f} ms'.format(
        np.mean(preprocessing_times[-300:])*1000, np.std(preprocessing_times[-300:])*1000))
    print('Average prediction time: {:.2f} ms ± {:.2f} ms'.format(
        np.mean(pred_times[-300:])*1000, np.std(pred_times[-300:])*1000))
    return seq_pred_times

if __name__ == '__main__':
    measure_run_time(model_path, test_path, seq_length, stations)



