# go through all images in test folder
# for each image, use model to predict label, if prediction is wrong check quality of image and add to quality list
# if quality is good, add to good list, if quality is bad, add to bad list
# if the image is good quality add the argmax value of the prediction to the good quality list, same for bad quality
# print out the mean value of the predictions of good quality images
# print out the mean value of the predictions of bad quality images
# print out the number of good and bad images
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2

from src.resources.config import get_stations_config
from src.utils.get_paths import get_frame_paths

# hypothesis one: with quality weights more of the misclassified images will be of poor quality
# hypothesis two: with quality weights the prediction value of argmax will be higher for good quality images

#with weights the model will perform more errors on the poor-quality images as it pays more attaention to the good quality images, also it should perform better on good quality images i.e. a bigger portion of correct classifications should be on good quality iamges
# with weights: makes more mistakes on poor quality images and less on good quality images
test_path_baseline = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/baseline/Levanger_and_StOlavs/test'
test_path_sequence = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/test'
df_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/baseline/Levanger_and_StOlavs/test_dirname_good_quality_df.csv'
df_path_sequence = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/test_dirname_good_quality_df.csv'
model_path_with_weights_baseline = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-01/20:33:48/best_model'
model_path_without_weights_baseline = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-04-13/22:31:04/best_model'
#model_path_with_weights_sequence = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-02/10:03:02/best_model'
model_path_with_weights_sequence = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-11/10:36:02/best_model'
#model_path_without_weights_sequence = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-02/16:08:35/best_model'
model_path_without_weights_sequence = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-09/12:07:18/best_model'


# Function to preprocess images
def preprocess_frame(frame_path):
    frame = tf.keras.utils.load_img(frame_path, color_mode='rgb', target_size=(224, 224))
    frame = tf.cast(frame, tf.float32)
    frame = frame / 127.5 - 1
    return frame

def preprocess_frame_baseline(frame_path):
    frame = tf.keras.utils.load_img(frame_path, color_mode='rgb', target_size=(224, 224), interpolation='bilinear')
    frame = tf.cast(frame, tf.float32)
    frame = frame / 127.5 - 1
    return frame

def load_image_sequence(frame_paths, seq_length):
    sequence = [preprocess_frame(frame_path) for frame_path in frame_paths]
    if len(sequence) != seq_length:
        # add zero padding to make the total equal seq_length
        zero_frame = tf.zeros_like(sequence[-1], dtype=tf.float32)
        num_repeats = seq_length - len(frame_paths)
        sequence = sequence + ([zero_frame] * num_repeats)
    sequence = tf.stack(sequence)
    return sequence


# Function to get predictions for a sequence of images
def get_predictions_frame(frame, model):
    frame = np.expand_dims(frame, axis=0)  # Model expects batch dimension
    predictions = model.predict(frame)
    return predictions[0]

def get_predictions_sequence(sequence, model):
    sequence = np.array(sequence)  # Convert list to np.array
    sequence = np.expand_dims(sequence, axis=0)  # Model expects batch dimension
    predictions = model.predict(sequence)
    return predictions[0]

def get_quality(df, frame_paths):
    qualities = [df[df['dirname'] == frame_path]['good_quality'].values[0] for frame_path in frame_paths]
    return qualities

def test_quality_weights_baseline(test_path, df_path, model_path, stations_config):
    df = pd.read_csv(df_path)
    model = tf.keras.models.load_model(model_path)
    stations = list(stations_config.keys())
    num_good_quality_correct = 0
    num_good_quality_incorrect = 0
    num_bad_quality_correct = 0
    num_bad_quality_incorrect = 0
    pred_values_good_quality_correct = []
    pred_values_good_quality_incorrect = []
    pred_values_bad_quality_correct = []
    pred_values_bad_quality_incorrect = []

    for station in sorted(os.listdir(test_path)):
        print(station)
        station_path = os.path.join(test_path, station)
        for frame in sorted(os.listdir(station_path)):
            frame_path = os.path.join(station_path, frame)
            quality = df[df['dirname'] == frame_path]['good_quality'].values[0]
            print(quality)
            preprocessed_frame = preprocess_frame_baseline(frame_path)
            pred = get_predictions_frame(preprocessed_frame, model)
            pred_station = stations[np.argmax(pred)]
            pred_value = np.max(pred)

            if pred_station != station: # If the prediction is wrong
                if quality == 1:
                    num_good_quality_incorrect += 1
                    pred_values_good_quality_incorrect.append(pred_value)
                else:
                    num_bad_quality_incorrect += 1
                    pred_values_bad_quality_incorrect.append(pred_value)

            else: # If the prediction is correct
                if quality == 1:
                    pred_values_good_quality_correct.append(pred_value)
                    num_good_quality_correct += 1
                else:
                    pred_values_bad_quality_correct.append(pred_value)
                    num_bad_quality_correct += 1


    print(f'Number of images: {num_good_quality_correct + num_good_quality_incorrect + num_bad_quality_correct + num_bad_quality_incorrect}')
    print(f'Number of good quality images: {num_good_quality_correct + num_good_quality_incorrect}')
    print(f'Number of bad quality images: {num_bad_quality_correct + num_bad_quality_incorrect}')
    print(f'Number of good quality images correctly classified: {num_good_quality_correct}')
    print(f'Number of good quality images incorrectly classified: {num_good_quality_incorrect}')
    print(f'Number of bad quality images correctly classified: {num_bad_quality_correct}')
    print(f'Number of bad quality images incorrectly classified: {num_bad_quality_incorrect}')
    print(f'Mean value of predictions for good quality images correct: {np.mean(pred_values_good_quality_correct)}')
    print(f'Mean value of predictions for bad quality images correct: {np.mean(pred_values_bad_quality_correct)}')
    print(f'Mean value of predictions for good quality images incorrect: {np.mean(pred_values_good_quality_incorrect)}')
    print(f'Mean value of predictions for bad quality images incorrect: {np.mean(pred_values_bad_quality_incorrect)}')


#test_quality_weights_baseline(test_path_baseline, df_path, model_path_with_weights_baseline, get_stations_config(3))

'''
Without quality weights:
Number of images: 3284
Number of good quality images: 1875
Number of bad quality images: 1409
Number of good quality images correctly classified: 1189
Number of good quality images incorrectly classified: 686
Number of bad quality images correctly classified: 603
Number of bad quality images incorrectly classified: 806
Mean value of predictions for good quality images correct: 0.7828441858291626
Mean value of predictions for bad quality images correct: 0.7092009782791138
Mean value of predictions for good quality images incorrect: 0.634597897529602
Mean value of predictions for bad quality images incorrect: 0.5865112543106079

percentage of misclassified images that are of bad quality: bad quality incorrect / total incorrect = 806 / (806 +686) = 0,5402144772
percentage of correctly classified images that are of good quality: good quality correct / total correct = 1189 / (1189 + 603) = 0,6635044643

mean prediction value for good quality correct: 0.7828441858291626 

With quality weights:
Number of images: 3284
Number of good quality images: 1875
Number of bad quality images: 1409
Number of good quality images correctly classified: 1162
Number of good quality images incorrectly classified: 713
Number of bad quality images correctly classified: 608
Number of bad quality images incorrectly classified: 801
Mean value of predictions for good quality images correct: 0.8061686754226685
Mean value of predictions for bad quality images correct: 0.7139179706573486
Mean value of predictions for good quality images incorrect: 0.655379056930542
Mean value of predictions for bad quality images incorrect: 0.6028760671615601

percentage of misclassified images that are of bad quality (should be higher here): bad quality incorrect / total incorrect =  801 / (801 + 713) = 0,5290620872
percentage of correctly classified images that are of good quality (should be higher here): good quality correct / total correct = 1162 / (1162 + 608) = 0,6564971751

mean prediction value for good quality correct: 0.8061686754226685


'''


def test_quality_weights_sequence(test_path, df_path, model_path, stations_config, seq_length):
    df = pd.read_csv(df_path)
    model = tf.keras.models.load_model(model_path)
    stations = list(stations_config.keys())
    num_good_quality_correct = 0
    num_good_quality_incorrect = 0
    num_bad_quality_correct = 0
    num_bad_quality_incorrect = 0
    pred_values_good_quality_correct = []
    pred_values_good_quality_incorrect = []
    pred_values_bad_quality_correct = []
    pred_values_bad_quality_incorrect = []

    for patient in sorted(os.listdir(test_path)):
        patient_path = os.path.join(test_path, patient)
        for station in sorted(os.listdir(patient_path)):
            station_path = os.path.join(patient_path, station)
            frame_paths = get_frame_paths(station_path, 'sequence')
            num_frames = len(frame_paths)
            sequences = [frame_paths[i: i + seq_length] for i in range(0, num_frames, seq_length)]
            for sequence in sequences:
                loaded_sequence = load_image_sequence(sequence, seq_length)
                prediction = get_predictions_sequence(loaded_sequence, model).tolist()
                pred_station = stations[np.argmax(prediction)]
                pred_value = np.max(prediction)
                qualities = get_quality(df, sequence)
                # find the average quality score for the sequence
                qualities = np.array(qualities)
                quality = np.mean(qualities)
                print(quality)

                if pred_station != station: # If the prediction is wrong
                    if quality > 0.5:
                        num_good_quality_incorrect += 1
                        pred_values_good_quality_incorrect.append(pred_value)
                    else:
                        num_bad_quality_incorrect += 1
                        pred_values_bad_quality_incorrect.append(pred_value)

                else: # If the prediction is correct
                    if quality > 0.5:
                        pred_values_good_quality_correct.append(pred_value)
                        num_good_quality_correct += 1
                    else:
                        pred_values_bad_quality_correct.append(pred_value)
                        num_bad_quality_correct += 1


    print(f'Number of sequences: {num_good_quality_correct + num_good_quality_incorrect + num_bad_quality_correct + num_bad_quality_incorrect}')
    print(f'Number of good quality sequences: {num_good_quality_correct + num_good_quality_incorrect}')
    print(f'Number of bad quality sequences: {num_bad_quality_correct + num_bad_quality_incorrect}')
    print(f'Number of good quality sequences correctly classified: {num_good_quality_correct}')
    print(f'Number of good quality sequences incorrectly classified: {num_good_quality_incorrect}')
    print(f'Number of bad quality sequences correctly classified: {num_bad_quality_correct}')
    print(f'Number of bad quality sequences incorrectly classified: {num_bad_quality_incorrect}')
    print(f'Mean value of predictions for good quality sequences correct: {np.mean(pred_values_good_quality_correct)}')
    print(f'Mean value of predictions for bad quality sequences correct: {np.mean(pred_values_bad_quality_correct)}')
    print(f'Mean value of predictions for good quality sequences incorrect: {np.mean(pred_values_good_quality_incorrect)}')
    print(f'Mean value of predictions for bad quality sequences incorrect: {np.mean(pred_values_bad_quality_incorrect)}')


test_quality_weights_sequence(test_path_sequence, df_path_sequence, model_path_with_weights_sequence, get_stations_config(3), 10)

"""
Without quality weights: 16:08:35
Number of sequences: 346
Number of good quality sequences: 199
Number of bad quality sequences: 147
Number of good quality sequences correctly classified: 136
Number of good quality sequences incorrectly classified: 63
Number of bad quality sequences correctly classified: 69
Number of bad quality sequences incorrectly classified: 78
Mean value of predictions for good quality sequences correct: 0.7958736266283428
Mean value of predictions for bad quality sequences correct: 0.6942202122747034
Mean value of predictions for good quality sequences incorrect: 0.7129678972183712
Mean value of predictions for bad quality sequences incorrect: 0.6201292482706217

percentage of misclassified images that are of bad quality: bad quality incorrect / total incorrect = 78 / (78 + 63) = 0,5531914894
percentage of correctly classified images that are of good quality: good quality correct / total correct = 136 / (136 + 69) = 0,6634146341

mean prediction value for good quality correct: 0.7958736266283428

with quality weights:
Number of sequences: 346
Number of good quality sequences: 199
Number of bad quality sequences: 147
Number of good quality sequences correctly classified: 123
Number of good quality sequences incorrectly classified: 76
Number of bad quality sequences correctly classified: 64
Number of bad quality sequences incorrectly classified: 83
Mean value of predictions for good quality sequences correct: 0.8812365025524201
Mean value of predictions for bad quality sequences correct: 0.7737961262464523
Mean value of predictions for good quality sequences incorrect: 0.7636641016916225
Mean value of predictions for bad quality sequences incorrect: 0.7121919939316899

percentage of misclassified images that are of bad quality (should be higher here): bad quality incorrect / total incorrect = 83 / (83 + 76) = 0,5220125786
percentage of correctly classified images that are of good quality (should be higher here): good quality correct / total correct = 123 / (123 + 64) = 0,6577540107

mean prediction value for good quality correct: 0.8812365025524201

------------------------------------------------------------------------------------------------------------

Without quality weights: 12:07:18
Number of sequences: 346
Number of good quality sequences: 199
Number of bad quality sequences: 147
Number of good quality sequences correctly classified: 149
Number of good quality sequences incorrectly classified: 50
Number of bad quality sequences correctly classified: 61
Number of bad quality sequences incorrectly classified: 86
Mean value of predictions for good quality sequences correct: 0.8017311646234269
Mean value of predictions for bad quality sequences correct: 0.7712601164325339
Mean value of predictions for good quality sequences incorrect: 0.572988497018814
Mean value of predictions for bad quality sequences incorrect: 0.6157517313610675

percentage of misclassified images that are of bad quality: bad quality incorrect / total incorrect = 86 / (86 + 50) = 0,6323529412
percentage of correctly classified images that are of good quality: good quality correct / total correct = 149 / (149 + 61) = 0,7095238095

mean prediction value for good quality correct: 0.8017311646234269

with quality weights: 10:36:02
Number of sequences: 346
Number of good quality sequences: 199
Number of bad quality sequences: 147
Number of good quality sequences correctly classified: 137
Number of good quality sequences incorrectly classified: 62
Number of bad quality sequences correctly classified: 73
Number of bad quality sequences incorrectly classified: 74
Mean value of predictions for good quality sequences correct: 0.843729804684646
Mean value of predictions for bad quality sequences correct: 0.7471797360132818
Mean value of predictions for good quality sequences incorrect: 0.7018487813972658
Mean value of predictions for bad quality sequences incorrect: 0.6900817596831837

percentage of misclassified images that are of bad quality (should be higher here): bad quality incorrect / total incorrect = 74 / (74 + 62) = 0,5441176471
percentage of correctly classified images that are of good quality (should be higher here): good quality correct / total correct = 137 / (137 + 73) = 0,6523809524

mean prediction value for good quality correct: 0.843729804684646

"""
