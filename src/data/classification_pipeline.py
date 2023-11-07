import logging
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random

from src.resources.augmentation import GammaTransform, ContrastScale, Blur, BrightnessTransform, GaussianShadow, \
    RandomAugmentation, Rotation, NonLinearMap
from src.utils.get_paths import get_station_paths, get_frame_paths

log = logging.getLogger()


# config_handler.set_global(bar='classic', spinner='classic')


class ClassificationPipeline:

    def __init__(self, data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment):
        self.data_path = data_path
        self.test_ds_path = test_ds_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.station_names = station_names
        self.num_stations = num_stations
        self.validation_split = validation_split
        self.test_split = test_split
        self.augment = augment
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None




class BaselineClassificationPipeline(ClassificationPipeline):

    def __init__(self, data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment):
        super().__init__(data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                         num_stations, augment)

    def loader_function(self):
        print(self.station_names)

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=os.path.join(self.data_path, 'train'),
            seed=123,
            labels='inferred',
            label_mode=self.get_label_mode(),
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=True
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=os.path.join(self.data_path, 'val'),
            labels='inferred',
            label_mode=self.get_label_mode(),
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=False
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.test_ds_path,
            labels='inferred',
            label_mode=self.get_label_mode(),
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=False
        )


        self.train_ds = self.prepare(train_ds, augment=self.augment)
        self.val_ds = self.prepare(val_ds, augment=False)  # no augmentation for validation and test data
        self.test_ds = self.prepare(test_ds, augment=False)

        '''
        # Used to test operations
        image_batch, labels_batch = next(iter(self.val_ds))
        first_image = image_batch[0]
        print('first image: ', first_image)
        print('min: ', np.min(first_image), 'max: ', np.max(first_image))
        print('image ', first_image)
        plt.figure(figsize=(10, 10))
        #first_image = (first_image + 1) / 2
        plt.imshow(first_image)
        plt.title('training' + str(labels_batch[0]))
        plt.show()
        '''




        return self.train_ds, self.val_ds, self.test_ds

    #TODO: remove
    '''
    def crop_image(self, image):
        cropped_image = tf.image.crop_to_bounding_box(image, 24, 71, 223, 150)
        return cropped_image
    '''

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

        rescale = tf.keras.Sequential([
            # tf.keras.layers.Rescaling(1. / 127.5, offset=-1)  # specific for mobilenet TODO: change for other models
            tf.keras.layers.Rescaling(1. / 255.0)
        ])
        '''
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),  # rotating by a random amount in the range [-10% * 2pi, 10% * 2pi]
            tf.keras.layers.RandomContrast(0.1)  # (x - mean) * contrast_factor + mean
        ])
        '''


        augmentation_layers = [GammaTransform(low=0.5, high=1.5), Blur(sigma_max=1.5), NonLinearMap()]
                               #Rotation(max_angle=30), GaussianShadow(sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.9), strength=(0.5, 0.8))]
        data_augmentation = RandomAugmentation(augmentation_layers)

        # Rescale all datasets.
        ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)

        if augment:
            # Use data augmentation only on the training set.
            #ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            #ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x, y: (tf.py_function(func=data_augmentation, inp=[x], Tout=tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)


class SequenceClassificationPipeline(ClassificationPipeline):
    def __init__(self, data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment, stations_config, seq_length, stride):
        super().__init__(data_path, test_ds_path, batch_size, image_shape, validation_split, test_split, station_names,
                         num_stations, augment)
        self.seq_length = seq_length
        self.stride = stride
        self.stations_config = stations_config
        self.train_paths, self.val_paths = self.get_training_paths()

    def get_training_paths(self):
        station_paths_list = get_station_paths(self.data_path)
        train_paths, val_paths = self.split_data(station_paths_list)
        return train_paths, val_paths

    def split_data(self, frames):

        # Split data using train_test_split
        total_samples = len(frames)
        validation_samples = int(self.validation_split * total_samples)

        indices = np.arange(total_samples)

        train_ind, val_ind = train_test_split(indices, test_size=validation_samples, random_state=123)

        # Index sequences and labels using the indices
        train_frames = [frames[i] for i in train_ind]
        val_frames = [frames[i] for i in val_ind]

        return train_frames, val_frames

    def load_image(self, image_path):
        """Load and preprocess a single image from a file path"""
        img = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=None)
        img = tf.cast(img, tf.float32)  # pixel values are in range [0, 255] to apply preprocessing layer in ml_models
        img = img[100:1035, 530:1658]  # Cropping the image to the region of interest
        #img = (tf.cast(img, tf.float32) / 127.5) - 1  # specific for mobilenet, inception TODO: change for other models
        #img = img / 255.0 # normalizing the image to be in range [0, 1] #TODO: change for other models
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
        station_path = station_path.numpy().decode('utf-8')  # converts station path from tf tensor to string
        frame_paths = get_frame_paths(station_path)  # gets frame paths for one station

        num_frames = len(frame_paths)
        if num_frames < (self.seq_length * self.stride):
            start_index = 0
        else:
            start_index = random.randint(0, num_frames - (self.seq_length * self.stride))
        end_index = start_index + (self.seq_length * self.stride)
        sequence_paths = frame_paths[start_index:end_index:self.stride]  # len(sequence_paths) = seq_length (or less)
        sequence = self.load_image_sequence(sequence_paths)
        #sequence = tf.convert_to_tensor(sequence)  # convert sequence to tf tensor
        #print('sequence_tensor: ', sequence)
        return sequence

    def get_label_from_path(self, path):
        station_folder = path.split('/')[-1]
        label = self.stations_config[station_folder]
        label_one_hot = tf.keras.utils.to_categorical(label, num_classes=self.num_stations)
        label_one_hot = tf.cast(label_one_hot, tf.float32)
        return label_one_hot

    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
    def gen(self, paths):
        while True:  # infinite generator
            patient = random.choice(paths)  # choose a random patient
            station = random.choice(patient)  # choose a random station
            yield station, self.get_label_from_path(station)

    def loader_function(self):
        # create dataset from generator (added lambda to provide a callable function to .from_generator)
        gen_train_ds = tf.data.Dataset.from_generator(lambda: self.gen(self.train_paths), output_shapes=(tf.TensorShape([]),
                                                                                                 tf.TensorShape([self.num_stations])),
                                                      output_types=(tf.string, tf.float32))

        gen_val_ds = tf.data.Dataset.from_generator(lambda: self.gen(self.val_paths), output_shapes=(tf.TensorShape([]),
                                                                                             tf.TensorShape([self.num_stations])),
                                                    output_types=(tf.string, tf.float32))

        # creates sequence of image paths for each station using tf.py_function
        train_ds = gen_train_ds.map(lambda x, y: (tf.py_function(func=self.create_sequence,
                                                                 inp=[x], Tout=tf.float32), y),
                                    num_parallel_calls=tf.data.AUTOTUNE)

        val_ds = gen_val_ds.map(lambda x, y: (tf.py_function(func=self.create_sequence,
                                                             inp=[x], Tout=tf.float32), y),
                                num_parallel_calls=tf.data.AUTOTUNE)

        '''
        # https://www.tensorflow.org/tutorials/images/data_augmentation
        data_augmentation = tf.keras.Sequential([
            #tf.keras.layers.RandomRotation(0.1),  # rotating by a random amount in the range [-10% * 2pi, 10% * 2pi]
            #tf.keras.layers.RandomContrast(0.1)  # (x - mean) * contrast_factor + mean
            # add custom data augmentation here using GammaTransform class
            GammaTransform()
        ])
        
        '''

        if self.augment:
            augmentation_layers = [GammaTransform(), ContrastScale(), Blur(), BrightnessTransform(), NonLinearMap(),
                                   GaussianShadow(sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.9), strength=(0.5, 0.8))]
            data_augmentation = RandomAugmentation(augmentation_layers)

            # apply data augmentation to training data
            train_ds = train_ds.map(lambda x, y: (tf.py_function(func=data_augmentation,
                                                                 inp=[x], Tout=tf.float32), y),
                                    num_parallel_calls=tf.data.AUTOTUNE)

        # batch and prefetch the datasets
        self.train_ds = train_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = val_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return self.train_ds, self.val_ds
