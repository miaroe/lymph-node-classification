import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import cv2

from src.resources.architectures.ml_models import get_arch
from src.resources.config import get_stations_config, get_num_stations
from src.resources.loss import get_loss
from src.visualization.confusion_matrix import confusion_matrix_and_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.resources.train_config import get_config
from src.utils.get_paths import get_frame_paths, get_test_station_paths
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape, LSTM, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential


def create_model(instance_size, num_stations, stateful):
    # Create the base model from the pre-trained model
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=instance_size, pooling=None)

    for layer in base_model.layers[:-11]:
        layer.trainable = False

    # Make sure the correct layers are frozen
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name, layer.trainable)

    # Create the input layer for the sequence of images
    sequence_input = Input(shape=(None, *instance_size))  # (B, T, H, W, C)

    # Apply the CNN base model to each image in the sequence
    x = TimeDistributed(base_model)(sequence_input)  # (B, T, H', W', C')

    # Apply Global Average Pooling to each frame in the sequence
    x = TimeDistributed(tf.keras.layers.GlobalMaxPooling2D())(x)  # (B, T, C')

    # Create an LSTM layer
    x = LSTM(64, return_sequences=True, stateful=stateful)(x)  # (B, T, lstm_output_dim)

    x = LSTM(64, return_sequences=False, stateful=stateful)(x)  # (B, lstm_output_dim)

    # Create a dense layer
    # x = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)  # (B, dense_output_dim)
    x = Dense(256, activation='relu')(x)  # (B, dense_output_dim)

    # Create a dropout layer
    x = Dropout(0.5)(x)  # (B, dense_output_dim)

    # Create the output layer for classification
    output = Dense(num_stations, activation='softmax')(x)  # (B, num_classes)

    # Create the combined model
    model = Model(inputs=sequence_input, outputs=output)

    return model

# Used to create a test dataset using data from Ålesund, needs to be cropped differently


def get_label_one_hot(station_folder, stations_config, num_stations):
    label = stations_config[station_folder]
    label_one_hot = tf.keras.utils.to_categorical(label, num_classes=num_stations)
    label_one_hot = tf.cast(label_one_hot, tf.float32)
    return label_one_hot

def get_path_labels(paths, stations_config, num_stations):
    labels = [path.split('/')[-1] for path in paths]
    labels = [label.split('_')[1] for label in labels] # split Station_11L_001 to 11L
    return [get_label_one_hot(label, stations_config, num_stations) for label in labels]

def preprocess_frames(image_path):
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    frame = frame[150:875, 500:1370]
    frame = tf.cast(frame, tf.float32)
    frame = tf.image.resize(frame, [224, 224])
    frame = (frame - 127.5) / 127.5
    return frame

def load_image_sequence(frame_paths, seq_length):
    sequence = [preprocess_frames(frame_path) for frame_path in frame_paths]
    if len(sequence) != seq_length:
        # add zero padding to make the total equal seq_length
        zero_frame = tf.zeros_like(sequence[-1], dtype=tf.float32)
        num_repeats = seq_length - len(frame_paths)
        sequence = sequence + ([zero_frame] * num_repeats)
    sequence = tf.stack(sequence)
    return sequence

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


# evaluate sequence model per sequence
def evaluate_sequence_model_per_seq(model_path, reports_path, seq_length, stations_config_nr, batch_size):
    print("Evaluating model: " + model_path)

    test_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Aalesund'
    stations_config = get_stations_config(stations_config_nr)
    num_stations = get_num_stations(stations_config_nr)
    test_station_paths = get_test_station_paths(test_path)

    sequences_list = []
    labels_list = []
    for station_path in test_station_paths:
        sequences = create_sequences_test(station_path, seq_length, False)
        print('station_path:', station_path)
        labels = get_path_labels([station_path] * len(sequences), stations_config, num_stations)
        sequences_list.extend(sequences)
        labels_list.extend(labels)

    print('num sequences:', len(sequences_list))
    print('num labels:', len(labels_list))

    test_ds = tf.data.Dataset.from_tensor_slices((sequences_list, labels_list))
    test_ds = test_ds.batch(batch_size)

    # print the value range of the pixels
    for images, labels in test_ds.take(1):
        print('images shape: ', images.shape)  # (batch_size, seq_length, 256, 256, 3)
        print('min:', tf.reduce_min(images))
        print('max:', tf.reduce_max(images))
    num_images = 0

    for i, (images, labels) in enumerate(test_ds.take(3)):
        print('images shape: ', images.shape)  # (1, seq_length, 256, 256, 3)
        print('labels shape: ', labels.shape)  # (1, 8)
        for seq in range(images.shape[0]):
            plt.figure(figsize=(10, 10))
            for image in range(seq_length):
                num_images += 1
                plt.subplot(4, 4, image + 1)
                #image_norm = images[seq][image]
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

    config_path = os.path.join(model_path, 'config.json')
    config = get_config(config_path)
    train_config = config["train_config"]

    model = get_arch(train_config.get('model_arch'), train_config.get('instance_size'),
                     train_config.get('num_stations'), stateful=False)
    model.compile(loss=get_loss(train_config.get('loss')), optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    model.load_weights(filepath=os.path.join(model_path, 'best_model')).expect_partial()

    #stateless_model = tf.keras.models.load_model(os.path.join(model_path, 'best_model'))
    #model = create_model(instance_size=(224, 224, 3), num_stations=8, stateful=False)
    #model.set_weights(stateless_model.get_weights())

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
    print('precision:', precision_score(true_labels, pred_labels, average='weighted'))
    print('recall:', recall_score(true_labels, pred_labels, average='weighted'))

    #confusion_matrix_and_report(true_labels, pred_labels, num_stations,
    #                            train_config.get('stations_config'),
    #                            reports_path, 'sequence_test_')


evaluate_sequence_model_per_seq(model_path='/home/miaroe/workspace/lymph-node-classification/output/models/2024-04-19/22:44:29',
                         reports_path='/home/miaroe/workspace/lymph-node-classification/reports/2024-04-19/22:44:29/',
                         seq_length=10, stations_config_nr=3, batch_size=1)


# plot image
#image_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Aalesund/20240419-102838/Station_11L_001/frame_0.png'
#image = cv2.imread(image_path)
#image = image[150:875, 500:1370] #[y1:y2, x1:x2]
#image = image[140:850, 440:1440]
#plt.imshow(image)
#plt.show()

