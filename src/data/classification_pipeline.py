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

        ])
        # Apply cropping
        ds = ds.map(lambda x, y: (self.crop_images(x), y), num_parallel_calls=AUTOTUNE)
        # Resize and rescale all datasets.
        ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(500)
        # Use data augmentation only on the training set.
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)
