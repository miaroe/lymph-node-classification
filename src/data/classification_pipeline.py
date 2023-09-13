import logging
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar, config_handler
from sklearn.model_selection import train_test_split
import random

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
    def __init__(self, data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment, shuffle, stations_config, seq_length):
        super().__init__(data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                         num_stations, augment, shuffle)
        self.seq_length = seq_length
        self.shift = 20
        self.stride = 1
        self.stations_config = stations_config
        self.train_paths, self.val_paths = self.get_paths()

    def get_paths(self):
        station_paths_list = self.get_station_paths()
        train_paths, val_paths = self.split_data(station_paths_list)
        return train_paths, val_paths

    def get_station_paths(self):
        station_paths_list = []
        for patient_folder in sorted(os.listdir(self.data_path)):
            patient_path = os.path.join(self.data_path, patient_folder)
            if os.path.isdir(patient_path):
                station_paths_list.append([])
                for station_folder in os.listdir(patient_path):
                    station_path = os.path.join(patient_path, station_folder)
                    if os.path.isdir(station_path):
                        station_paths_list[-1].append(station_path)

        return station_paths_list

    def get_frame_paths(self, station_path):
        frame_paths_list = []
        frame_names = os.listdir(station_path)
        sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].replace('.png', '')))
        for frame in sorted_frame_names:
            frame_path = os.path.join(station_path, frame)
            if os.path.isfile(frame_path):
                frame_paths_list.append(frame_path)
        return frame_paths_list

    def split_data(self, frames):

        # Split data using train_test_split
        total_samples = len(frames)
        validation_samples = int(self.validation_split * total_samples)

        indices = np.arange(total_samples)

        train_ind, val_ind = train_test_split(indices, test_size=validation_samples, random_state=42)

        # Index sequences and labels using the indices
        train_frames = [frames[i] for i in train_ind]
        val_frames = [frames[i] for i in val_ind]

        return train_frames, val_frames

    def load_image(self, image_path):
        """Load and preprocess a single image from a file path"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)  # Assuming RGB images
        img = img[100:1035, 530:1658]  # Cropping the image to the region of interest
        img = tf.cast(img, tf.float32)
        # img = (tf.cast(img, tf.float32) / 127.5) - 1  # specific for mobilenet, inception TODO: change for other models
        img = tf.image.resize(img, self.image_shape)  # resizing the image to the desired shape
        return img

    def load_image_sequence(self, frame_paths):
        sequence = [self.load_image(frame_path) for frame_path in frame_paths]
        if len(sequence) != self.seq_length:
            # add zero padding to make the total equal seq_length
            zero_frame = np.zeros_like(sequence[-1], dtype=np.float32)
            num_repeats = self.seq_length - len(frame_paths)
            sequence = sequence + ([zero_frame] * num_repeats)
        sequence = tf.stack(sequence)
        return sequence

    # Function to create image sequences of seq_length for one station for one patient
    def create_sequence(self, station_path):
        station_path = station_path.numpy().decode('utf-8')  # convert station path from tf tensor to string
        frame_paths = self.get_frame_paths(station_path)  # get frame paths for one station

        num_frames = len(frame_paths)
        if num_frames < self.seq_length:
            start_index = 0
        else:
            start_index = random.randint(0, num_frames - (self.seq_length * self.stride))
        # frame_paths_ds = tf.data.Dataset.from_tensor_slices(frame_paths[start_index:])
        # frame_paths_sequence_ds = self.make_window_dataset(frame_paths_ds)
        end_index = start_index + (self.seq_length * self.stride)
        sequence_paths = frame_paths[start_index:end_index:self.stride]
        sequence = self.load_image_sequence(sequence_paths)
        sequence = tf.convert_to_tensor(sequence)  # convert sequence to tf tensor
        return sequence

    def get_label_from_path(self, path):
        station_folder = path.split('/')[-1]
        label = self.stations_config[station_folder]
        label_one_hot = tf.keras.utils.to_categorical(label, num_classes=self.num_stations)
        label_one_hot = tf.cast(label_one_hot, tf.float32)
        return label_one_hot

    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
    def gen(self):
        while True:  # infinite generator
            patient = random.choice(self.train_paths)  # choose a random patient
            station = random.choice(patient)  # choose a random station
            yield station, self.get_label_from_path(station)

    def loader_function(self):
        # create dataset from generator
        gen_train_ds = tf.data.Dataset.from_generator(self.gen, output_shapes=(tf.TensorShape([]),
                                                                               tf.TensorShape([self.num_stations])),
                                                      output_types=(tf.string, tf.float32))

        gen_val_ds = tf.data.Dataset.from_generator(self.gen, output_shapes=(tf.TensorShape([]),
                                                                             tf.TensorShape([self.num_stations])),
                                                    output_types=(tf.string, tf.float32))

        # create sequence of image paths for each station using tf.py_function
        train_ds = gen_train_ds.map(lambda x, y: (tf.py_function(func=self.create_sequence,
                                                                 inp=[x], Tout=tf.float32), y),
                                    num_parallel_calls=tf.data.AUTOTUNE)

        val_ds = gen_val_ds.map(lambda x, y: (tf.py_function(func=self.create_sequence,
                                                             inp=[x], Tout=tf.float32), y),
                                num_parallel_calls=tf.data.AUTOTUNE)

        '''
        # https://www.tensorflow.org/tutorials/images/data_augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),  # rotating by a random amount in the range [-10% * 2pi, 10% * 2pi]
            tf.keras.layers.RandomContrast(0.1)  # (x - mean) * contrast_factor + mean
        ])

        # apply data augmentation to training data
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        '''

        # shuffle and batch the datasets
        self.train_ds = train_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = val_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return self.train_ds, self.val_ds

