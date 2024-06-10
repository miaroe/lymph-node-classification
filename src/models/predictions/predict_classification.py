import shutil
import tensorflow as tf
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall
import h5py

from src.resources.architectures.ml_models import get_arch
from src.resources.config import get_stations_config, get_num_stations
from src.resources.loss import get_loss
from src.visualization.confusion_matrix import confusion_matrix_and_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.resources.train_config import get_config
from src.utils.get_paths import get_frame_paths, get_test_station_paths

dirname_test_df = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/test_dirname_label_df.csv'
model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/2024-06-01/19:51:30'
reports_path = '/home/miaroe/workspace/lymph-node-classification/reports/2024-06-04/21:28:00/'
model_name = 'best_model'
test_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/test'
data_path_cv = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence_cv'
stations_config_nr = 3

#------------------ Helper functions ------------------#

def get_frame_label_dict_modified(full_video_path):
    frame_label_dict = {}
    labels_path = os.path.join(full_video_path, 'labels.csv')
    # read the csv file using pandas
    df = pd.read_csv(labels_path, sep=',')

    # create a dictionary to store the mappings
    for index, row in df.iterrows():
        frame_index = (row['frame_number'])
        frame_label_dict[frame_index] = row['label']

    return frame_label_dict

def rescale_image(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image

def preprocess_image(image):
    image = tf.cast(image, tf.float32)
    image = image[100:1035, 530:1658]
    image = tf.image.resize(image, [256, 256], method='nearest')
    image = image / 255.0
    return image


#------------------ Server function ------------------#

def predict_classification_per_station(dirname_test_df, model_path, model_name, labels):
    model = tf.keras.models.load_model(os.path.join(model_path, model_name))
    dirname_test_df = pd.read_csv(dirname_test_df, sep=',')
    # for each row in the df, get the station and the dirname. Then for each dirname, loop through the images and predict.
    # Then take the average of the prediction arrays and assign the label to the station.
    # Then compare the label to the true label and calculate the accuracy.

    # create a df to store the predictions, including all columns from the original df
    station_predictions = []

    # loop through the df and get the dirname and station
    for index, row in dirname_test_df.iterrows():
        dirname = row['dirname']
        station = row['label']
        patient_id = row['patient_id']
        print('dirname:', dirname)

        prediction_per_image_arr = []
        num_correct_threshold = 0
        num_frames = 0

        # loop through the images in the dirname and predict
        for image_name in os.listdir(dirname):
            if image_name.endswith(".png"):
                img = tf.keras.utils.load_img(os.path.join(dirname, image_name), color_mode='rgb',
                                              target_size=None)
                img = preprocess_image(img)
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                prediction_per_image = model.predict(img_array)
                prediction_per_image_arr.append(prediction_per_image)
                num_frames += 1


        prediction_per_station_arr = np.mean(prediction_per_image_arr, axis=0)
        #print('prediction_per_station:', prediction_per_station_arr)
        prediction_value = np.round(100 * np.max(prediction_per_station_arr[0]), 2)
        #print('prediction_value:', prediction_value)
        prediction_station = labels[np.argmax(prediction_per_station_arr)]
        #print('prediction_station:', prediction_station)
        #print('station:', station)

        for prediction in prediction_per_image_arr:
            if np.max(prediction) > 0.5:
                prediction_station = labels[np.argmax(prediction)]
                if prediction_station == station:
                    num_correct_threshold += 1

        if prediction_station != station:
            print('prediction_station:', prediction_station)
            print('station:', station)
            print('prediction_value:', prediction_value)
            print('prediction_per_station:', prediction_per_station_arr)
            print('-----------------------------')


        # add the prediction to the station_predictions df
        station_predictions.append({'dirname': dirname, 'patient_id': patient_id, 'station': station,
                                                          'prediction_values_arr': prediction_per_station_arr,
                                                          'prediction_value': prediction_value,
                                                          'prediction_station': prediction_station,
                                                          'num_correct_threshold': num_correct_threshold / num_frames})

    station_predictions_df = pd.DataFrame(station_predictions)

    # save in model_path
    station_predictions_df.to_csv(os.path.join(model_path, 'station_predictions.csv'), index=False)

    # create confusion matrix
    stations_config = get_stations_config(stations_config_nr)
    true_labels = station_predictions_df['station']
    true_labels = [true_labels[i] for i in range(len(true_labels))]
    predicted_labels = station_predictions_df['prediction_station']
    predicted_labels = [predicted_labels[i] for i in range(len(predicted_labels))]

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'Accuracy: {accuracy:}')

    # Calculate precision
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    print(f'Precision: {precision:}')

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print(f'Recall: {recall:}')

    # use stations config to get the numbers of the labels
    for i in range(len(true_labels)):
        true_labels[i] = stations_config[true_labels[i]]
        predicted_labels[i] = stations_config[predicted_labels[i]]
    print('true_labels:', true_labels)
    print('predicted_labels:', predicted_labels)
    confusion_matrix_and_report(true_labels, predicted_labels, len(labels), stations_config, reports_path, 'avg_')


    #print('num_correct:', len(station_predictions_df[station_predictions_df['station'] == station_predictions_df['prediction_station']]))
    #print('num_total:', len(station_predictions_df))
    print('percent_correct:', len(station_predictions_df[station_predictions_df['station'] == station_predictions_df['prediction_station']]) / len(station_predictions_df) * 100)

#predict_classification_per_station(dirname_test_df, model_path, model_name, labels=list(get_stations_config(stations_config_nr).keys()))


# ------------------ Sequence ------------------#

def get_label_one_hot(station_folder, stations_config, num_stations):
    label = stations_config[station_folder]
    label_one_hot = tf.keras.utils.to_categorical(label, num_classes=num_stations)
    label_one_hot = tf.cast(label_one_hot, tf.float32)
    return label_one_hot

def get_path_labels(paths, stations_config, num_stations):
    labels = [path.split('/')[-1] for path in paths]
    return [get_label_one_hot(label, stations_config, num_stations) for label in labels]

def preprocess_frames(image_path):
    frame = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(224, 224))
    frame = tf.cast(frame, tf.float32)
    frame = frame / 127.5 - 1
    #frame = frame / 255.0
    return frame

def load_image_sequence(frame_paths, seq_length):
    sequence = [preprocess_frames(frame_path) for frame_path in frame_paths]
    if len(sequence) != seq_length:
        # add zero padding to make the total equal seq_length
        zero_frame = np.zeros_like(sequence[-1], dtype=np.float32)
        num_repeats = seq_length - len(frame_paths)
        sequence = sequence + ([zero_frame] * num_repeats)
    sequence = tf.stack(sequence)
    return sequence

def load_masked_image_multi_input(image_path):
    with h5py.File(image_path, 'r') as file:
        image = file['image'][:]
        mask = file['mask'][:]

        # resize to 224x224
        image = tf.image.resize(image, [224, 224], method='nearest')
        mask = tf.image.resize(mask, [224, 224], method='nearest')

        image = np.repeat(image, 3, axis=-1) # repeat grayscale image to create 3 channels
        return image, mask

def load_image_sequence_multi_input(frame_paths, seq_length):
    images_sequence = []
    masks_sequence = []
    for frame_path in frame_paths:
        image, mask = load_masked_image_multi_input(frame_path)
        images_sequence.append(image)
        masks_sequence.append(mask)

    # Ensure sequences are padded to the desired length
    while len(images_sequence) < seq_length:
        images_sequence.append(np.zeros((224, 224, 3), dtype=np.float32))
        masks_sequence.append(np.zeros((224, 224, 3), dtype=np.float32))

    images_sequence = tf.stack(images_sequence)
    masks_sequence = tf.stack(masks_sequence)
    return images_sequence, masks_sequence

def get_predictions(sequence, model):
    sequence = np.array(sequence)  # Convert list to np.array
    sequence = np.expand_dims(sequence, axis=0)  # Model expects batch dimension
    predictions = model.predict(sequence)
    return predictions[0]

def create_sequences_test(station_path, seq_length, convert_from_tensor):
    if convert_from_tensor:
        station_path = station_path.numpy().decode('utf-8')
    frame_paths = get_frame_paths(station_path, 'sequence')  # gets frame paths for one station
    num_frames = len(frame_paths)
    sequence_paths = [frame_paths[i: i + seq_length] for i in range(0, num_frames, seq_length)]
    sequences = [load_image_sequence(sequence, seq_length) for sequence in sequence_paths]
    return sequences

def create_sequences_test_multi_input(station_path, seq_length, convert_from_tensor):
    if convert_from_tensor:
        station_path = station_path.numpy().decode('utf-8')
    print('station_path:', station_path)
    frame_paths = get_frame_paths(station_path, 'sequence_with_segmentation')  # gets frame paths for one station
    num_frames = len(frame_paths)
    sequence_paths = [frame_paths[i: i + seq_length] for i in range(0, num_frames, seq_length)]
    image_sequences = []
    mask_sequences = []
    for sequence in sequence_paths:
        images, masks = load_image_sequence_multi_input(sequence, seq_length)
        image_sequences.append(images)
        mask_sequences.append(masks)
    return image_sequences, mask_sequences

def predict_sequence(model_path, model_name, test_path, seq_length):
    model = tf.keras.models.load_model(os.path.join(model_path, model_name))
    sequence_predictions = []
    true_labels = []
    for patient in os.listdir(test_path):
        patient_path = os.path.join(test_path, patient)
        for station in os.listdir(patient_path):
            station_path = os.path.join(patient_path, station)
            frame_paths = get_frame_paths(station_path, 'sequence')
            num_frames = len(frame_paths)
            sequences = [frame_paths[i: i + seq_length] for i in range(0, num_frames, seq_length)]
            for sequence in sequences:
                loaded_sequence = load_image_sequence(sequence, seq_length)
                loaded_sequence = loaded_sequence / 127.5 - 1
                prediction = get_predictions(loaded_sequence, model).tolist()
                sequence_predictions.append(np.argmax(prediction))
                true_labels.append(get_stations_config(stations_config_nr)[station])

    confusion_matrix_and_report(true_labels, sequence_predictions, 8, get_stations_config(stations_config_nr), reports_path, 'test_')
    return sequence_predictions, true_labels

#print(predict_sequence(model_path, model_name, test_path, seq_length=10))


# evaluate sequence model per station video --> mean accuracy per station
def evaluate_sequence_model(model_path, reports_path, seq_length, stations_config_nr, batch_size):
    print("Evaluating model: " + model_path)

    test_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/test'
    stations_config = get_stations_config(stations_config_nr)
    num_stations = get_num_stations(stations_config_nr)
    test_paths = get_test_station_paths(test_path, 'sequence')
    print('num test paths:', len(test_paths))
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, get_path_labels(test_paths, stations_config, num_stations)))
    test_ds = test_ds.map(lambda x, y: (tf.py_function(func=create_sequences_test, inp=[x, seq_length, True], Tout=tf.float32), y))
    test_ds = test_ds.batch(batch_size)
    '''
    #plotting
    plt.style.use('ggplot')
    # print the value range of the pixels
    for images, labels in test_ds.take(1):
        print('images shape: ', images.shape)  # (batch_size, seq_length, 256, 256, 3)
        print('min:', tf.reduce_min(images))
        print('max:', tf.reduce_max(images))
    num_images = 0

    for i, (images, labels) in enumerate(test_ds.take(3)):
        print('images shape: ', images.shape)  # (4, seq_length, 256, 256, 3)
        print('labels shape: ', labels.shape)  # (4, 8)
        for seq in range(images.shape[0]):
            plt.figure(figsize=(10, 10))
            for image in range(seq_length):
                num_images += 1
                plt.subplot(4, 4, image + 1)
                image_norm = (images[seq][image] + 1) * 127.5
                #image_norm = (images[seq][image]) * 255.0
                plt.imshow(image_norm.numpy().astype("uint8"))
                plt.title(f"Frame {image}, Label: {np.argmax(labels.numpy()[seq])}",
                          fontsize=10)
                plt.axis("off")
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.suptitle(f"Batch {i}, Sequence {seq}", fontsize=16)
            plt.show()
    print(f"Total images: {num_images}")
    '''

    config_path = os.path.join(model_path, 'config.json')
    config = get_config(config_path)
    train_config = config["train_config"]

    model = get_arch(train_config.get('model_arch'), train_config.get('instance_size'),
                     train_config.get('num_stations'), stateful=False)
    model.compile(loss=get_loss(train_config.get('loss')), optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    model.load_weights(filepath=os.path.join(model_path, 'best_model')).expect_partial()

    score_test = model.evaluate(test_ds, return_dict=True, steps=None)
    with open(os.path.join(reports_path, 'test_metrics.txt'), 'a') as f:
        f.write(f'\n\nTest metrics for model: {model_path}\n')
        f.write(f'{"Metric":<12}{"Value"}\n')
        for metric, value in score_test.items():
            f.write(f'{metric:<12}{value:<.4f}\n')

    #debug
    # Initialize the true labels and predicted labels arrays
    true_labels = []
    pred_labels = []

    for batch in test_ds:
        images, labels = batch  # shape=(4, seq_length, 224, 224, 3)
        pred_probs = model.predict(images)
        batch_pred_labels = np.argmax(pred_probs, axis=1)
        true_labels.extend(np.argmax(labels, axis=1))
        pred_labels.extend(batch_pred_labels)
    print('true_labels:', true_labels)
    print('pred_labels:', pred_labels)
    print('accuracy:', accuracy_score(true_labels, pred_labels))

    confusion_matrix_and_report(true_labels, pred_labels, num_stations,
                                train_config.get('stations_config'),
                                reports_path, 'sequence_test_')


#evaluate_sequence_model(model_path='/home/miaroe/workspace/lymph-node-classification/output/models/2024-04-24/15:20:54',
#                         reports_path='/home/miaroe/workspace/lymph-node-classification/reports/2024-04-24/15:20:54',
#                         seq_length=20, stations_config_nr=3, batch_size=4)


# evaluate sequence model per sequence
def evaluate_sequence_model_per_seq(model_path, reports_path, seq_length, stations_config_nr, batch_size):
    print("Evaluating model: " + model_path)

    test_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/test'
    stations_config = get_stations_config(stations_config_nr)
    num_stations = get_num_stations(stations_config_nr)
    test_station_paths = get_test_station_paths(test_path, 'sequence')

    sequences_list = []
    labels_list = []
    for station_path in test_station_paths:
        sequences = create_sequences_test(station_path, seq_length, False)
        labels = get_path_labels([station_path] * len(sequences), stations_config, num_stations)
        sequences_list.extend(sequences)
        labels_list.extend(labels)

    print('num sequences:', len(sequences_list))
    print('num labels:', len(labels_list))

    test_ds = tf.data.Dataset.from_tensor_slices((sequences_list, labels_list))
    test_ds = test_ds.batch(batch_size)

    # print the value range of the pixels
    for images, labels in test_ds.take(1):
        print('images shape: ', images.shape)  # (batch_size, seq_length, 224, 224, 3)
        print('min:', tf.reduce_min(images))
        print('max:', tf.reduce_max(images))

    config_path = os.path.join(model_path, 'config.json')
    config = get_config(config_path)
    train_config = config["train_config"]

    model = get_arch(train_config.get('model_arch'), train_config.get('instance_size'),
                     train_config.get('num_stations'), stateful=False)
    model.compile(loss=get_loss(train_config.get('loss')), optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    model.load_weights(filepath=os.path.join(model_path, 'best_model')).expect_partial()

    score_test = model.evaluate(test_ds, return_dict=True, steps=None)
    os.makedirs(reports_path, exist_ok=True)
    if not os.path.exists(os.path.join(reports_path, 'test_metrics.txt')):
        with open(os.path.join(reports_path, 'test_metrics.txt'), 'w') as f:
            f.write('Test metrics\n')
    with open(os.path.join(reports_path, 'test_metrics.txt'), 'a') as f:
        f.write(f'\n\nTest metrics for model: {model_path}\n')
        f.write(f'{"Metric":<12}{"Value"}\n')
        for metric, value in score_test.items():
            f.write(f'{metric:<12}{value:<.4f}\n')

    #debug
    # Initialize the true labels and predicted labels arrays
    true_labels = []
    pred_labels = []

    for batch in test_ds:
        images, labels = batch  # shape=(4, seq_length, 224, 224, 3)
        pred_probs = model.predict(images)
        batch_pred_label = np.argmax(pred_probs, axis=1)
        true_label = np.argmax(labels, axis=1)
        true_labels.extend(true_label)
        pred_labels.extend(batch_pred_label)

    print('true_labels:', true_labels)
    print('pred_labels:', pred_labels)
    print('accuracy:', accuracy_score(true_labels, pred_labels))

    confusion_matrix_and_report(true_labels, pred_labels, num_stations,
                                train_config.get('stations_config'),
                                reports_path, 'sequence_test_')


#evaluate_sequence_model_per_seq(model_path='/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-09/12:07:18',
#                         reports_path='/home/miaroe/workspace/lymph-node-classification/reports/2024-05-09/12:07:18/',
#                         seq_length=10, stations_config_nr=3, batch_size=1)


# evaluate multi input model per sequence
def evaluate_multi_input_model_per_seq(model_path, reports_path, seq_length, stations_config_nr, batch_size):
    print("Evaluating model: " + model_path)

    test_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence_segmentation/Levanger_and_StOlavs/test'
    stations_config = get_stations_config(stations_config_nr)
    num_stations = get_num_stations(stations_config_nr)
    test_station_paths = get_test_station_paths(test_path, 'sequence')

    image_sequences_list = []
    mask_sequences_list = []
    labels_list = []
    for station_path in test_station_paths:
        image_sequences, mask_sequences = create_sequences_test_multi_input(station_path, seq_length, False)
        print('image_sequences:', len(image_sequences))
        print('mask_sequences:', len(mask_sequences))
        labels = get_path_labels([station_path] * len(image_sequences), stations_config, num_stations)
        labels_list.extend(labels)
        image_sequences_list.extend(image_sequences)
        mask_sequences_list.extend(mask_sequences)


    print('num sequences:', len(image_sequences_list))
    print('num labels:', len(labels_list))

    test_ds = tf.data.Dataset.from_tensor_slices(((image_sequences_list, mask_sequences_list), labels_list))

    test_ds = test_ds.map(lambda x, y: (((x[0] - 0.5) * 2.0, x[1]), y), num_parallel_calls=tf.data.AUTOTUNE)

    test_ds = test_ds.batch(batch_size)

    for data in test_ds.take(1):
        images, labels = data

        # images is first channel and masks is second and third channel
        image_input = images[0]
        mask_input = images[1]

        print('image_input shape:', image_input.shape)
        print('mask_input shape:', mask_input.shape)
        print('labels shape:', labels.shape)


    config_path = os.path.join(model_path, 'config.json')
    config = get_config(config_path)
    train_config = config["train_config"]

    model = get_arch(train_config.get('model_arch'), train_config.get('instance_size'),
                     train_config.get('num_stations'), stateful=False)
    model.compile(loss=get_loss(train_config.get('loss')), optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    model.load_weights(filepath=os.path.join(model_path, 'best_model')).expect_partial()

    score_test = model.evaluate(test_ds, return_dict=True, steps=None)
    with open(os.path.join(reports_path, 'test_metrics.txt'), 'a') as f:
        f.write(f'\n\nTest metrics for model: {model_path}\n')
        f.write(f'{"Metric":<12}{"Value"}\n')
        for metric, value in score_test.items():
            f.write(f'{metric:<12}{value:<.4f}\n')

    #debug
    # Initialize the true labels and predicted labels arrays
    true_labels = []
    pred_labels = []

    for batch in test_ds:
        images, labels = batch  # shape=(4, seq_length, 224, 224, 3)
        pred_probs = model.predict(images)
        batch_pred_labels = np.argmax(pred_probs, axis=1)
        true_labels.extend(np.argmax(labels, axis=1))
        pred_labels.extend(batch_pred_labels)
    print('true_labels:', true_labels)
    print('pred_labels:', pred_labels)
    print('accuracy:', accuracy_score(true_labels, pred_labels))

    confusion_matrix_and_report(true_labels, pred_labels, num_stations,
                                train_config.get('stations_config'),
                                reports_path, 'sequence_test_')


#evaluate_multi_input_model_per_seq(model_path='/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-10/12:13:08',
#                         reports_path='/home/miaroe/workspace/lymph-node-classification/reports/2024-05-10/12:13:08/',
#                         seq_length=10, stations_config_nr=3, batch_size=1)



#------------------ Cross validation ------------------#

def evaluate_cv(n_splits):
    # calculate metrics for each fold and save average +- std to reports path
    accuracy_list = []
    precision_list = []
    recall_list = []

    for fold in range(n_splits):
        reports_path_fold = os.path.join(reports_path, f'fold_{fold}_v2')

        with open(os.path.join(reports_path_fold, 'test_metrics.txt'), 'r') as f:
            lines = f.readlines()
            accuracy = float(lines[5].split()[-1])
            precision = float(lines[6].split()[-1])
            recall = float(lines[7].split()[-1])

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)

    # calculate average and std for each metric
    avg_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list)
    avg_precision = np.mean(precision_list)
    std_precision = np.std(precision_list)
    avg_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)


    with open(os.path.join(reports_path, 'test_metrics.txt'), 'a') as f:
        f.write(f'\n\nAverage metrics for {n_splits} folds\n')
        f.write(f'{"Metric":<12}{"Value"}\n')
        f.write(f'{"Accuracy":<12}{avg_accuracy:<.4f} +- {std_accuracy:<.4f}\n')
        f.write(f'{"Precision":<12}{avg_precision:<.4f} +- {std_precision:<.4f}\n')
        f.write(f'{"Recall":<12}{avg_recall:<.4f} +- {std_recall:<.4f}\n')

#evaluate_cv(n_splits=5)

# write a function that retrieves accuracy, precision and recall from each class from sequence_report.csv from each fold and calculates average and std
def evaluate_cv_per_class(n_splits, report_folder, name):
    accuracy_list = []
    precision_dict = {}
    recall_dict = {}

    for fold in range(n_splits):
        file_path = os.path.join(report_folder, f'fold_{fold}_v2', f'{name}_report.csv')
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        print(df)

        # Extract overall accuracy
        accuracy_list.append(df.loc['accuracy']['precision'])

        for class_label in df.index[:-3]:  # Skipping 'accuracy', 'macro avg', 'weighted avg'
            if class_label not in precision_dict:
                precision_dict[class_label] = []
                recall_dict[class_label] = []

            precision_dict[class_label].append(df.loc[class_label]['precision'])
            recall_dict[class_label].append(df.loc[class_label]['recall'])

    # Calculate averages and stds
    results = {
        'accuracy': {
            'average': np.mean(accuracy_list),
            'std': np.std(accuracy_list)
        }
    }

    for class_label in precision_dict:
        results[class_label] = {
            'precision': {
                'average': np.mean(precision_dict[class_label]),
                'std': np.std(precision_dict[class_label])
            },
            'recall': {
                'average': np.mean(recall_dict[class_label]),
                'std': np.std(recall_dict[class_label])
            }
        }

    # Save results to file
    with open(os.path.join(report_folder, f'{name}_metrics.txt'), 'w') as f:
        f.write(f'{name} metrics\n')
        f.write(f'{"Accuracy":<12}{results["accuracy"]["average"]:<12.4f} +- {results["accuracy"]["std"]:<12.4f}\n\n')
        f.write(f'{"Class":<12}{"Precision":<28}{"Recall"}\n')
        for class_label in results:
            if class_label == 'accuracy':
                continue
            f.write(f'{class_label:<12}{results[class_label]["precision"]["average"]:<12.4f} +- {results[class_label]["precision"]["std"]:<12.4f}{results[class_label]["recall"]["average"]:<12.4f} +- {results[class_label]["recall"]["std"]:<12.4f}\n')


evaluate_cv_per_class(n_splits=5, report_folder=reports_path, name='sequence_stateful')





