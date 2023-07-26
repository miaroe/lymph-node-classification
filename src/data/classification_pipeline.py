import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger()

class EBUSClassificationPipeline:

    def __init__(self, data_path, batch_size, image_shape, validation_split, station_names, num_stations, augment, stations_config):
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.station_names = station_names
        self.num_stations = num_stations
        self.validation_split = validation_split
        self.augment = augment
        self.stations_config = stations_config
        self.train_ds = None
        self.val_ds = None

    def loader_function(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # adapted from https://www.tensorflow.org/tutorials/images/transfer_learning and
        # https://www.tensorflow.org/tutorials/load_data/images
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1), #rotating by a random amount in the range [-10% * 2pi, 10% * 2pi]
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomContrast(0.1) #(x - mean) * contrast_factor + mean
        ])

        resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.Resizing(256, 256),
            tf.keras.layers.Rescaling(1. / 255)
        ])

        print(self.station_names)

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.data_path,
            validation_split=self.validation_split,
            subset='training',
            seed=123,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=True
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.data_path,
            validation_split=self.validation_split,
            subset='validation',
            seed=123,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=True
        )

        #Apply cropping and then resize and normalize the images
        train_ds = train_ds.map(lambda x, y: (self.crop_images(x), y), num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y), num_parallel_calls=AUTOTUNE)

        #Used to test operations
        '''
        image_batch, labels_batch = next(iter(train_ds))
        first_image = image_batch[0]
        print(np.min(first_image), np.max(first_image))

        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                augmented_image = data_augmentation(images[i])
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_image)
                plt.axis("off")
            plt.show()
        '''
        train_ds = train_ds.cache().shuffle(1000)
        val_ds = val_ds.cache()

        if self.augment:
            train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)

        return train_ds.prefetch(buffer_size=AUTOTUNE), val_ds.prefetch(buffer_size=AUTOTUNE)

    def crop_images(self, image):
        cropped_image = tf.image.crop_to_bounding_box(image, 24, 71, 223, 150)
        return cropped_image