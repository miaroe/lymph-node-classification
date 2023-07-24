import logging
import tensorflow as tf

from src.resources import train_config

log = logging.getLogger()


class EBUSClassificationPipeline:

    def __init__(self, data_path, batch_size, image_shape, validation_split, station_names, num_stations):
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.station_names = station_names
        self.num_stations = num_stations
        self.validation_split = validation_split
        self.train_ds = None
        self.val_ds = None

    def loader_function(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        print(self.station_names)

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.data_path,
            validation_split=self.validation_split,
            subset='training',
            seed=123,
            labels='inferred',
            #label_mode='binary',
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
            #label_mode='binary',
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=True
        )
        #TODO: need to crop images?
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds
