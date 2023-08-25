import logging
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar, config_handler
from sklearn.model_selection import train_test_split

from src.data.full_video_label_dict import get_frame_label_dict

log = logging.getLogger()


# config_handler.set_global(bar='classic', spinner='classic')


class ClassificationPipeline:

    def __init__(self, data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment,
                 shuffle):
        self.data_path = data_path
        self.test_ds_path = test_ds_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.station_names = station_names
        self.num_stations = num_stations
        self.validation_split = validation_split
        self.test_split = test_split
        self.augment = augment
        self.shuffle = shuffle
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def loader_function(self):
        print(self.station_names)

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.data_path,
            validation_split=self.validation_split,
            subset='training',
            seed=123,
            labels='inferred',
            label_mode=self.get_label_mode(),
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=self.shuffle
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.data_path,
            validation_split=self.validation_split,
            subset='validation',
            seed=123,
            labels='inferred',
            label_mode=self.get_label_mode(),
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=self.shuffle
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.test_ds_path,
            seed=123,
            labels='inferred',
            label_mode=self.get_label_mode(),
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=self.shuffle
        )

        self.train_ds = self.prepare(train_ds, augment=self.augment)
        self.val_ds = self.prepare(val_ds, augment=False)  # no augmentation for validation and test data
        self.test_ds = self.prepare(test_ds, augment=False)

        # Used to test operations
        '''
        image_batch, labels_batch = next(iter(train_ds))
        first_image = image_batch[0]
        print(np.min(first_image), np.max(first_image))
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i])
                plt.axis("off")
            plt.show()
        '''

        return self.train_ds, self.val_ds, self.test_ds

    def crop_image(self, image):
        cropped_image = tf.image.crop_to_bounding_box(image, 24, 71, 223, 150)
        return cropped_image

    def get_label_mode(self):
        if self.num_stations > 2:
            label_mode = 'categorical'
        else:
            label_mode = 'int'
        return label_mode

    def prepare(self, ds, augment=False):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # adapted from https://www.tensorflow.org/tutorials/images/transfer_learning,
        # https://www.tensorflow.org/tutorials/load_data/images
        # https://www.tensorflow.org/tutorials/images/data_augmentation

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),  # rotating by a random amount in the range [-10% * 2pi, 10% * 2pi]
            tf.keras.layers.RandomContrast(0.1)  # (x - mean) * contrast_factor + mean
        ])

        resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.Resizing(256, 256),
            tf.keras.layers.Rescaling(1. / 127.5, offset=-1)  # specific for mobilenet TODO: change for other models
            # tf.keras.layers.Rescaling(1. / 255.0)

        ])
        # Apply cropping
        ds = ds.map(lambda x, y: (self.crop_image(x), y), num_parallel_calls=AUTOTUNE)
        # Resize and rescale all datasets.
        ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

        # Use data augmentation only on the training set.
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)


class BaselineClassificationPipeline(ClassificationPipeline):

    def __init__(self, data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment, shuffle):
        super().__init__(data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                         num_stations, augment, shuffle)

    def loader_function(self):
        self.train_ds, self.val_ds, self.test_ds = super().loader_function()
        return self.train_ds, self.val_ds, self.test_ds


class SequenceClassificationPipeline(ClassificationPipeline):
    def __init__(self, data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names, num_stations,
                 augment, shuffle, stations_config, seq_length):
        super().__init__(data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names, num_stations,
                         augment, shuffle)
        self.seq_length = seq_length
        self.stations_config = stations_config

    # Function to create image sequences of seq_length for one station for one patient
    # input frames have the shape (batch_size, 256, 256, 3) and labels have the shape (batch_size, num_stations)
    # NB: batch_size for the last batch might be smaller than self.batch_size
    def create_sequence_for_station(self, frames, seq_length):
        num_frames = len(frames)
        if num_frames >= seq_length:
            # If there are more than or equal to 10 frames, extract frames with an equal interval
            step_size = num_frames // seq_length
            indices = tf.range(start=0, limit=num_frames, delta=step_size)[:seq_length]

        else:
            # If there are less than 10 frames, repeat the last frame to make the total 10
            last_frame = frames[-1]
            num_repeats = seq_length - num_frames
            frames = frames + [last_frame] * num_repeats
            indices = tf.range(start=0, limit=seq_length, delta=1)
        return [frames[i] for i in indices]

    def load_image(self, image_path):
        """Load and preprocess a single image from a file path"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)  # Assuming RGB images
        img = img[100:1035, 530:1658]
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, self.image_shape)

        # Data augmentation TODO: check if this is correct, will it augment validation images as well?
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),  # rotating by a random amount in the range [-10% * 2pi, 10% * 2pi]
            tf.keras.layers.RandomContrast(0.1)  # (x - mean) * contrast_factor + mean
        ])
        img = data_augmentation(img, training=True)
        return img



    """
    This function creates sequences of images for each station for each patient.
    The sequences are created by taking frames at equal intervals from the station folder, 
    so there is one sequence of 10 frames for each station folder in each patient folder.
    """
    def create_sequences(self):
        sequences = []
        labels = []
        with alive_bar(len(os.listdir(self.data_path)), title='Loading data', bar='bubbles', spinner='fishes') as bar:
            for patient_folder in sorted(os.listdir(self.data_path)):  # add sorted for same order
                print('Loading patient: ', patient_folder)
                patient_folder_path = os.path.join(self.data_path, patient_folder)
                station_folders = os.listdir(patient_folder_path)
                for station_folder in station_folders:
                    print('Loading station: ', station_folder)
                    station_path = os.path.join(patient_folder_path, station_folder)
                    frame_names = os.listdir(station_path)
                    sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].replace('.png', '')))
                    frame_paths = [os.path.join(station_path, frame) for frame in sorted_frame_names]
                    frames = [self.load_image(frame_path) for frame_path in frame_paths]
                    sequence = self.create_sequence_for_station(frames, self.seq_length)
                    sequences.append(sequence)
                    label = self.stations_config[station_folder]
                    labels.append(tf.keras.utils.to_categorical(label, num_classes=self.num_stations))
                bar()

        # Split data using train_test_split
        total_samples = len(sequences)
        validation_samples = int(self.validation_split * total_samples)
        test_samples = int(self.test_split * total_samples)

        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=(validation_samples + test_samples), random_state=1, stratify=labels)

        # TODO: import test_ds from folder
        #test_ds = None

        return (X_train, y_train), (X_val, y_val)

    def loader_function(self):

        # preprocess data and split into training, validation and test datasets
        train, val = self.create_sequences()

        # create tf datasets
        train_ds = tf.data.Dataset.from_tensor_slices(train)
        val_ds = tf.data.Dataset.from_tensor_slices(val)
        # test_ds = tf.data.Dataset.from_tensor_slices(test)

        # shuffle and batch datasets
        self.train_ds = train_ds.shuffle(len(train)).batch(self.batch_size)
        self.val_ds = val_ds.batch(self.batch_size)
        # self.test_ds = test_ds.batch(self.batch_size)

        return self.train_ds, self.val_ds


'''
#-----------------------------------------------------------------------------------------------------------------------
    def loader_function(self, dir):

        # Load and preprocess all video sequences
        videos_data = []
        videos_labels = []
        for patient_folder in sorted(os.listdir(self.data_path)):
            print('Loading patient: ', patient_folder)
            print('batch_size ', self.batch_size)
            train_ds, val_ds, test_ds = super().loader_function(os.path.join(self.data_path, patient_folder))
            print('train_ds ', train_ds)
            train_ds = self.create_sequences_for_dataset(train_ds, self.seq_length)
            val_ds = self.create_sequences_for_dataset(val_ds, self.seq_length)

        return self.train_ds, self.val_ds

    # Function to create image sequences of seq_length for one station for one patient
    # input frames have the shape (batch_size, 256, 256, 3) and labels have the shape (batch_size, num_stations)
    # NB: batch_size for the last batch might be smaller than self.batch_size
    def create_sequence_for_station(self, frames, seq_length):
        num_frames = len(frames)
        print(frames)
        print('num_frames ', num_frames)
        if num_frames >= seq_length:
            # If there are more than or equal to 10 frames, extract frames with an equal interval
            step_size = num_frames // seq_length
            indices = tf.range(start=0, limit=num_frames, delta=step_size)[:seq_length]

        else:
            # If there are less than 10 frames, repeat the last frame to make the total 10
            last_frame = frames[-1]
            num_repeats = seq_length - num_frames
            frames = frames + [last_frame] * num_repeats
            indices = tf.range(start=0, limit=seq_length, delta=1)
            print('indices ', indices)
        return [frames[i] for i in indices]

    # Function to create image sequences for all classes in the dataset
    def create_sequences_for_dataset(self, dataset, seq_length):
        sequences = []
        for frames, label in dataset:
            seq = self.create_sequence_for_station(frames, seq_length)
            sequences.append((seq, label))
        return sequences
'''
