import os
import tensorflow as tf
import time
import numpy as np
import sys
sys.path.append('/home/miaroe/workspace/lymph-node-classification')

from src.resources.config import get_stations_config

model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-06-03/15:28:20/fold_0_v2'
test_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/FullVideos/EBUS_Levanger_full_videos/Patient_001/Sequence_001'
seq_length = 10
stations = list(get_stations_config(3).keys())

# Function to preprocess images
def preprocess_frame(frame_path):
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

def load_model_run_time(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def get_frame_paths_run_time(station_path):
    frame_paths_list = []
    frame_names = os.listdir(station_path)
    frame_names = [name for name in frame_names if not name.endswith('.csv')]
    sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].replace('.png', '')))
    print(sorted_frame_names)
    for frame in sorted_frame_names:
        frame_path = os.path.join(station_path, frame)
        if os.path.isfile(frame_path):
            frame_paths_list.append(frame_path)
    return frame_paths_list
def measure_run_time(model_path, test_path, sequence_length, stations):

    model = load_model_run_time(os.path.join(model_path, 'best_model'))
    sequence = []
    seq_counter = 0
    seq_pred_times = []
    preprocessing_times = []
    pred_times = []
    frame_paths = get_frame_paths_run_time(test_path)
    start_time = time.time()
    for i in range(11):
        for frame in frame_paths:
            preprocessing_start = time.time()
            frame = preprocess_frame(frame)
            preprocessing_end = time.time()

            sequence.append(frame)  # Add the frame to the sequence

            if len(sequence) == sequence_length:
                # Get predictions for the current sequence
                pred_time_start = time.time()
                pred = get_predictions(sequence, model)
                pred_time_end = time.time()
                end_time = time.time()
                #prediction = stations[np.argmax(pred)]
                sequence = []
                #print(prediction)
                if i > 0:
                    seq_pred_times.append(end_time - start_time)
                    pred_times.append(pred_time_end - pred_time_start)
                    preprocessing_times.append(preprocessing_end - preprocessing_start)
                start_time = time.time()

                seq_counter += 1
                if seq_counter == 50:
                    break

    print('Average sequence prediction time in ms: {:.2f} ± {:.2f} ms'.format(
        np.mean(seq_pred_times)*1000, np.std(seq_pred_times)*1000))
    print('Average sequence preprocessing time: {:.2f} ms ± {:.2f} ms'.format(
        np.mean(preprocessing_times)*1000, np.std(preprocessing_times)*1000))
    print('Average prediction time: {:.2f} ms ± {:.2f} ms'.format(
        np.mean(pred_times)*1000, np.std(pred_times)*1000))

    return seq_pred_times

if __name__ == '__main__':
    measure_run_time(model_path, test_path, seq_length, stations)

'''
Article 
(2024-06-03/15:28:20/fold_0_v2)
Average sequence prediction time in ms: 679.05 ± 51.81 ms
Average sequence preprocessing time: 59.74 ms ± 6.32 ms
Average prediction time: 76.96 ms ± 21.18 ms

Average sequence prediction time in ms: 652.31 ± 34.92 ms
Average sequence preprocessing time: 57.16 ms ± 3.22 ms
Average prediction time: 70.90 ms ± 19.22 ms

Average sequence prediction time in ms: 651.59 ± 27.35 ms
Average sequence preprocessing time: 56.95 ms ± 2.49 ms
Average prediction time: 69.47 ms ± 14.72 ms
'''



