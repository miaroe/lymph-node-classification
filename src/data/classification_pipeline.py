import logging
import tensorflow as tf
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import h5py

from src.resources.augmentation import GammaTransform, ContrastScale, Blur, GaussianShadow, \
    RandomAugmentation, Rotation, RandomAugmentationSequence
from src.utils.get_paths import get_training_station_paths, get_frame_paths, get_baseline_station_paths, \
    get_quality_dataframes

log = logging.getLogger()


class ClassificationPipeline:

    def __init__(self, data_path, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment, stations_config, model_arch):
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.station_names = station_names
        self.num_stations = num_stations
        self.validation_split = validation_split
        self.test_split = test_split
        self.augment = augment
        self.stations_config = stations_config
        self.model_arch = model_arch
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def get_label_one_hot(self, station_folder):
        label = self.stations_config[station_folder]
        label_one_hot = tf.keras.utils.to_categorical(label, num_classes=self.num_stations)
        label_one_hot = tf.cast(label_one_hot, tf.float32)
        return label_one_hot

    def transform_quality_scores(self, quality_scores):
        weight_for_high_quality = 3.0
        weight_for_low_quality = 1.0
        quality_scores = np.where(quality_scores == 1, weight_for_high_quality, weight_for_low_quality)
        return tf.cast(quality_scores, tf.float32)


class Baseline(ClassificationPipeline):

    def __init__(self, data_path, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment, stations_config, model_arch):
        super().__init__(data_path, batch_size, image_shape, validation_split, test_split, station_names,
                         num_stations, augment, stations_config, model_arch)

    def loader_function(self):
        print(self.station_names)

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=os.path.join(self.data_path, 'train'),
            #seed=42,
            labels='inferred',
            label_mode=self.get_label_mode(),
            color_mode='rgb', # 'grayscale' or 'rgb' (1 or 3 channels)
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
            directory=os.path.join(self.data_path, 'test'),
            labels='inferred',
            label_mode=self.get_label_mode(),
            color_mode='rgb',
            class_names=self.station_names,
            batch_size=self.batch_size,
            image_size=self.image_shape,
            shuffle=False
        )


        self.train_ds = self.prepare(train_ds, augment=self.augment)
        self.val_ds = self.prepare(val_ds)  # no augmentation or shuffling for validation and test data
        self.test_ds = self.prepare(test_ds)

        # Used to test operations
        image_batch, labels_batch = next(iter(self.train_ds))
        first_image = image_batch[0]
        print('first image: ', first_image)
        print('min: ', np.min(first_image), 'max: ', np.max(first_image))
        plt.figure(figsize=(10, 10))
        first_image = first_image / 255.0
        plt.imshow(first_image)
        plt.title('train' + str(labels_batch[0]))
        plt.show()

        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(3):
            for i in range(32):
                ax = plt.subplot(6, 6, i + 1)
                plt.imshow(images[i] / 255.0)
                plt.title(self.station_names[np.argmax(labels[i])])
                plt.axis("off")
            plt.show()


        return self.train_ds, self.val_ds, self.test_ds

    def get_label_mode(self):
        if self.num_stations > 2:
            label_mode = 'categorical'
        else:
            label_mode = 'int'
        return label_mode

    def data_augmentation(self, x):
        augmentation_layers = [GammaTransform(low=0.5, high=1.5), Blur(sigma_max=1.0), Rotation(max_angle=30),
                               ContrastScale(min_scale=0.5, max_scale=1.5),
                               GaussianShadow(sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.9), strength=(0.5, 0.8))]
        return RandomAugmentation(augmentation_layers)(x)


    @tf.function
    def data_augmentation_map(self, x, y):
        augmented_x = tf.py_function(func=self.data_augmentation, inp=[x], Tout=tf.float32)
        return augmented_x, y

    @tf.function
    def data_augmentation_map_with_quality(self, x, y, z):
        augmented_x = tf.py_function(func=self.data_augmentation, inp=[x], Tout=tf.float32)
        return augmented_x, y, z

    def prepare(self, ds, augment=False):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # adapted from https://www.tensorflow.org/tutorials/images/transfer_learning,
        # https://www.tensorflow.org/tutorials/load_data/images
        # https://www.tensorflow.org/tutorials/images/data_augmentation

        rescale = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255.0)
        ])

        # Rescale all datasets.
        ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)

        if augment:
            # Use data augmentation only on the training set.
            ds = ds.unbatch() # augmentation is applied to each image separately
            ds = ds.map(self.data_augmentation_map, num_parallel_calls=AUTOTUNE)
            ds = ds.batch(self.batch_size)

        if self.model_arch == 'resnet' or self.model_arch == 'vgg16' or self.model_arch == 'mobilenetV3Small':
            # scale to [0, 255]
            ds = ds.map(lambda x, y: (x * 255., y), num_parallel_calls=AUTOTUNE)

        elif self.model_arch == 'inception' or self.model_arch == 'mobilenetV2':
            # scale to [-1, 1]
            ds = ds.map(lambda x, y: (x * 2. - 1., y), num_parallel_calls=AUTOTUNE)

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)


# ----------------- Loader with quality -----------------

    def loader_function_with_quality(self):
        train_paths, val_paths, test_paths = get_baseline_station_paths(self.data_path)
        train_quality_df, val_quality_df, test_quality_df = get_quality_dataframes(self.data_path)
        print(self.get_path_qualities(train_paths, train_quality_df))

        train_ds = tf.data.Dataset.from_tensor_slices(
            (train_paths, self.get_path_labels(train_paths), self.get_path_qualities(train_paths, train_quality_df)))
        val_ds = tf.data.Dataset.from_tensor_slices(
            (val_paths, self.get_path_labels(val_paths), self.get_path_qualities(val_paths, val_quality_df)))
        test_ds = tf.data.Dataset.from_tensor_slices(
            (test_paths, self.get_path_labels(test_paths), self.get_path_qualities(test_paths, test_quality_df)))

        # shuffle train_ds
        train_ds = train_ds.shuffle(buffer_size=len(train_paths), reshuffle_each_iteration=True)

        train_ds = train_ds.map(self.load_image_with_label_and_quality)
        val_ds = val_ds.map(self.load_image_with_label_and_quality)
        test_ds = test_ds.map(self.load_image_with_label_and_quality)

        self.train_ds = self.prepare_with_quality(train_ds, augment=True)
        self.val_ds = self.prepare_with_quality(val_ds)
        self.test_ds = self.prepare_with_quality(test_ds)

        '''
        # Used to test operations
        image_batch, labels_batch, qualities_batch = next(iter(self.train_ds))
        first_image = image_batch[0]
        print('first image: ', first_image)
        print('min: ', np.min(first_image), 'max: ', np.max(first_image))
        plt.figure(figsize=(10, 10))
        # normalize from [-1,1] to [0,1]
        first_image = (first_image + 1.) / 2.
        plt.imshow(first_image)
        plt.title('train' + str(labels_batch[0]))
        plt.show()
        j = 0

        for images, labels, qualities in self.train_ds.take(4):
            plt.figure(figsize=(10, 10))
            for i in range(32):
                ax = plt.subplot(6, 6, i + 1)
                # normalize from [-1,1] to [0,1]
                plt.imshow((images[i] + 1.) / 2.)
                plt.title(self.station_names[np.argmax(labels[i])] + ' ' + str(qualities[i].numpy()))
                plt.axis("off")
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Adjust top spacing for suptitle
            plt.suptitle(f"Batch {j}")
            plt.show()
            j += 1
        '''

        return self.train_ds, self.val_ds, self.test_ds

    def get_path_labels(self, paths):
        labels = [os.path.basename(os.path.dirname(path)) for path in paths]
        labels_one_hot = [self.get_label_one_hot(label) for label in labels]
        return labels_one_hot

    def get_path_qualities(self, paths, quality_df):
        qualities = [quality_df[quality_df['dirname'] == path]['good_quality'].values[0] for path in paths]
        qualities = self.transform_quality_scores(np.array(qualities))
        return qualities

    def load_image_tf(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, self.image_shape)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def load_image_with_label_and_quality(self, path, label, quality_score):
        img = self.load_image_tf(path)
        return img, label, quality_score

    def prepare_with_quality(self, ds, augment=False):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        if augment:
            ds = ds.map(self.data_augmentation_map_with_quality, num_parallel_calls=AUTOTUNE)

        if self.model_arch == 'resnet' or self.model_arch == 'vgg16' or self.model_arch == 'mobilenetV3Small':
            # scale to [0, 255]
            ds = ds.map(lambda x, y: (x * 255., y), num_parallel_calls=AUTOTUNE)

        elif self.model_arch == 'inception' or self.model_arch == 'mobilenetV2':
            # scale to [-1, 1]
            ds = ds.map(lambda x, y: (x * 2. - 1., y), num_parallel_calls=AUTOTUNE)

        ds = ds.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        return ds


class Sequence(ClassificationPipeline):
    def __init__(self, data_path, model_type, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment, stations_config, seq_length, set_stride, model_arch, instance_size, full_video, use_gen):
        super().__init__(data_path, batch_size, image_shape, validation_split, test_split, station_names,
                         num_stations, augment, stations_config, model_arch)
        self.model_type = model_type
        self.seq_length = seq_length
        self.set_stride = set_stride
        self.instance_size = instance_size
        self.full_video = full_video
        self.use_gen = use_gen
        self.train_paths, self.val_paths = get_training_station_paths(self.data_path)
        self.train_quality, self.val_quality = get_quality_dataframes(self.data_path, test=False)


    def get_path_labels(self, paths):
        labels = [path.split('/')[-1] for path in paths]
        return [self.get_label_one_hot(label) for label in labels]


    def get_quality(self, frame_paths, ds_type):
        if ds_type == 'train':
            qualities = [self.train_quality[self.train_quality['dirname'] == frame_path]['good_quality'].values[0] for frame_path in frame_paths]
        else:
            qualities = [self.val_quality[self.val_quality['dirname'] == frame_path]['good_quality'].values[0] for frame_path in frame_paths]
        return self.transform_quality_scores(np.array(qualities))



    def load_image(self, image_path):
        """Load and preprocess a single image from a file path"""
        img = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=self.image_shape)
        img = tf.cast(img, tf.float32)
        img = img / 255.0 # normalizing the image to be in range [0, 1] for augmentation
        #img = tf.expand_dims(img, axis=-1) # add channel dimension if color_mode='grayscale'
        return img

    def load_image_sequence(self, frame_paths):
        sequence = [self.load_image(frame_path) for frame_path in frame_paths] # numpy list of tf tensors
        if len(sequence) != self.seq_length:
            # add zero padding to make the total equal seq_length
            zero_frame = np.zeros_like(sequence[-1], dtype=np.float32)
            num_repeats = self.seq_length - len(frame_paths)
            sequence = sequence + ([zero_frame] * num_repeats)
        sequence = tf.stack(sequence)
        return sequence

    def load_image_full_video(self, image_path):
        # Decode the image only if image_path is not '0'
        def _load_image():
            img = tf.io.read_file(image_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, self.image_shape)
            img = img / 255.0  # Normalize the image to be in range [0, 1]
            return img

        # Return a zero tensor if image_path is '0'
        def _zero_image():
            return tf.zeros(self.image_shape + (3,), dtype=tf.float32)

        # Use tf.cond to choose the operation conditionally
        return tf.cond(tf.equal(image_path, tf.constant('0')), _zero_image, _load_image)

    def load_image_sequence_full_video(self, frame_paths):
        # Convert frame_paths to a tensor of strings
        frame_paths_tensor = tf.convert_to_tensor(frame_paths, dtype=tf.string)
        # Use tf.map_fn to apply load_image to each element in frame_paths_tensor
        sequence = tf.map_fn(self.load_image_full_video, frame_paths_tensor, dtype=tf.float32)
        return sequence


    # Function to create image sequences of seq_length for one station for one patient
    def create_sequence(self, station_path):
        station_path = station_path.numpy().decode('utf-8')  # converts station path from tf tensor to string
        frame_paths = get_frame_paths(station_path, self.model_type)  # gets frame paths for one station
        num_frames = len(frame_paths)
        if self.set_stride: stride = random.randint(1, 3)
        else: stride = 1
        if num_frames < (self.seq_length * stride):
            start_index = 0
        else:
            start_index = random.randint(0, num_frames - (self.seq_length * stride))
        end_index = start_index + (self.seq_length * stride)
        sequence_paths = frame_paths[start_index:end_index:stride]  # len(sequence_paths) = seq_length (or less)
        sequence = self.load_image_sequence(sequence_paths)
        return sequence

    # loading qualities before creating the sequence due to eager execution error when passing dictionary to map function
    def create_sequence_with_quality(self, station_path, label, ds_label):
        station_path = station_path.numpy().decode('utf-8')  # converts station path from tf tensor to string
        frame_paths = get_frame_paths(station_path, self.model_type)  # gets frame paths for one station
        num_frames = len(frame_paths)
        if self.set_stride: stride = random.randint(1, 3)
        else: stride = 1
        if num_frames < (self.seq_length * stride):
            start_index = 0
        else:
            start_index = random.randint(0, num_frames - (self.seq_length * stride))
        end_index = start_index + (self.seq_length * stride)
        sequence_paths = frame_paths[start_index:end_index:stride]  # len(sequence_paths) = seq_length (or less)
        qualities = self.get_quality(sequence_paths, ds_label)
        # find the average quality score for the sequence
        qualities = np.array(qualities)
        quality = np.mean(qualities)
        sequence = self.load_image_sequence(sequence_paths)
        return sequence, label, quality

    # Function to create multiple image sequences of seq_length for one station for one patient
    def create_sequences_for_ds(self, ds_paths):
        sequences_list = []
        labels_list = []

        for station_path in ds_paths:
            label = self.get_label_one_hot(station_path.split('/')[-1])
            frame_paths = get_frame_paths(station_path, self.model_type)

            num_frames = len(frame_paths)
            for i in range(0, num_frames, self.seq_length):
                sequence = frame_paths[i: i + self.seq_length]
                if len(sequence) != self.seq_length:
                    num_repeats = self.seq_length - len(sequence)
                    sequence = sequence + (['0'] * num_repeats)  # add zero padding to make the total equal seq_length
                sequences_list.append(sequence)
                labels_list.append(label)
        print('number of sequences:', len(sequences_list))

        # Convert sequences and labels to TensorFlow datasets
        sequences_ds = tf.data.Dataset.from_tensor_slices(sequences_list)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels_list)
        # Use map with num_parallel_calls for efficient loading and processing
        sequences_ds = sequences_ds.map(
            lambda x: tf.py_function(func=self.load_image_sequence_full_video, inp=[x], Tout=tf.float32),
            num_parallel_calls=tf.data.AUTOTUNE)
        return tf.data.Dataset.zip((sequences_ds, labels_ds))


    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
    def gen(self, paths):
        while True:  # infinite generator
            #choose a random station from the list of stations
            station = random.choice(list(self.stations_config.keys()))
            # choose a random patient from the list of patients for that station
            filtered_paths = [path for path in paths if path.endswith(station)]
            patient = random.choice(filtered_paths)

            yield patient, self.get_label_one_hot(station)


    def data_augmentationSeq(self, x):
        augmentation_layers = [GammaTransform(low=0.5, high=1.5), Blur(sigma_max=1.0), Rotation(max_angle=30),
                               ContrastScale(min_scale=0.5, max_scale=1.5),
                               GaussianShadow(sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.9), strength=(0.5, 0.8))]
        #augmentation_layers = [Blur(sigma_max=0.1)]
        return RandomAugmentationSequence(augmentation_layers)(x)

    @tf.function
    def data_augmentation_map(self, x, y):
        augmented_x = tf.py_function(func=self.data_augmentationSeq, inp=[x], Tout=tf.float32)
        return augmented_x, y

    @tf.function
    def data_augmentation_map_with_quality(self, x, y, z):
        augmented_x = tf.py_function(func=self.data_augmentationSeq, inp=[x], Tout=tf.float32)
        return augmented_x, y, z


    def loader_function(self):
        if self.full_video:
            train_ds = self.create_sequences_for_ds(self.train_paths)
            val_ds = self.create_sequences_for_ds(self.val_paths)

            # shuffle train_ds
            train_ds = train_ds.shuffle(buffer_size=256, reshuffle_each_iteration=True)

        else:
            if self.use_gen:
                # create dataset from generator (added lambda to provide a callable function to .from_generator)
                train_ds = tf.data.Dataset.from_generator(lambda: self.gen(self.train_paths), output_shapes=(tf.TensorShape([]),
                                                                                                         tf.TensorShape([self.num_stations])),
                                                              output_types=(tf.string, tf.float32))

                val_ds = tf.data.Dataset.from_generator(lambda: self.gen(self.val_paths), output_shapes=(tf.TensorShape([]),
                                                                                                     tf.TensorShape([self.num_stations])),
                                                        output_types=(tf.string, tf.float32))

            else:
                train_ds = tf.data.Dataset.from_tensor_slices((self.train_paths, self.get_path_labels(self.train_paths)))
                val_ds = tf.data.Dataset.from_tensor_slices((self.val_paths, self.get_path_labels(self.val_paths)))

                print('len train paths:', len(self.train_paths))

                # shuffle train_ds
                train_ds = train_ds.shuffle(buffer_size=len(self.train_paths), reshuffle_each_iteration=True)

            # creates sequence of image paths for each station using tf.py_function
            train_ds = train_ds.map(lambda x, y: (tf.py_function(func=self.create_sequence,
                                                                     inp=[x], Tout=tf.float32), y),
                                        num_parallel_calls=tf.data.AUTOTUNE)

            val_ds = val_ds.map(lambda x, y: (tf.py_function(func=self.create_sequence,
                                                                 inp=[x], Tout=tf.float32), y),
                                    num_parallel_calls=tf.data.AUTOTUNE)

            if self.augment:
                # apply data augmentation to training data
                train_ds = train_ds.map(self.data_augmentation_map, num_parallel_calls=tf.data.AUTOTUNE)

            if self.model_arch in ['resnet-lstm', 'mobileNetV3Small-lstm']:
                # scale each image back to [0, 255]
                print("scaling images to [0, 255]")
                train_ds = train_ds.map(lambda x, y: (x * 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
                val_ds = val_ds.map(lambda x, y: (x * 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)

            if self.model_arch in ['inception-lstm', 'mobilenetV2-lstm']:
                # scale from [0, 1] to [-1, 1]
                train_ds = train_ds.map(lambda x, y: ((x - 0.5) * 2.0, y), num_parallel_calls=tf.data.AUTOTUNE)
                val_ds = val_ds.map(lambda x, y: ((x - 0.5) * 2.0, y), num_parallel_calls=tf.data.AUTOTUNE)


        # batch and prefetch the datasets
        self.train_ds = train_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = val_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return self.train_ds, self.val_ds

    def loader_function_with_quality(self):
        train_ds = tf.data.Dataset.from_tensor_slices((self.train_paths, self.get_path_labels(self.train_paths), tf.ones(len(self.train_paths))))
        val_ds = tf.data.Dataset.from_tensor_slices((self.val_paths, self.get_path_labels(self.val_paths), tf.ones(len(self.val_paths))))

        train_ds = train_ds.shuffle(buffer_size=len(self.train_paths), reshuffle_each_iteration=True)

        # creates sequence of image paths for each station using tf.py_function
        train_ds = train_ds.map(lambda x, y, z: (tf.py_function(func=self.create_sequence_with_quality,
                                                               inp=[x, y, 'train'], Tout=(tf.float32, tf.float32, tf.float32))),
                                 num_parallel_calls=tf.data.AUTOTUNE)

        val_ds = val_ds.map(lambda x, y, z: (tf.py_function(func=self.create_sequence_with_quality,
                                                           inp=[x, y, 'val'], Tout=(tf.float32, tf.float32, tf.float32))),
                             num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment:
            # apply data augmentation to training data
            train_ds = train_ds.map(self.data_augmentation_map_with_quality, num_parallel_calls=tf.data.AUTOTUNE)

        if self.model_arch in ['resnet-lstm', 'mobileNetV3Small-lstm']:
            # scale each image back to [0, 255]
            train_ds = train_ds.map(lambda x, y, z: (x * 255.0, y, z), num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(lambda x, y, z: (x * 255.0, y, z), num_parallel_calls=tf.data.AUTOTUNE)

        if self.model_arch in ['inception-lstm', 'mobilenetV2-lstm']:
            # scale from [0, 1] to [-1, 1]
            train_ds = train_ds.map(lambda x, y, z: ((x - 0.5) * 2.0, y, z), num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(lambda x, y, z: ((x - 0.5) * 2.0, y, z), num_parallel_calls=tf.data.AUTOTUNE)


        def set_shapes(images, labels, qualities):
            images.set_shape([None, 224, 224, 3])
            labels.set_shape([8])
            qualities.set_shape([])
            return images, labels, qualities

        train_ds = train_ds.map(set_shapes)
        val_ds = val_ds.map(set_shapes)

        # batch and prefetch the datasets
        self.train_ds = train_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = val_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return self.train_ds, self.val_ds


class SequenceWithSegmentation(ClassificationPipeline):
    def __init__(self, data_path, model_type, batch_size, image_shape, validation_split, test_split, station_names,
                 num_stations, augment, stations_config, seq_length, set_stride, model_arch, instance_size):
        super().__init__(data_path, batch_size, image_shape, validation_split, test_split, station_names,
                         num_stations, augment, stations_config, model_arch)
        self.model_type = model_type
        self.seq_length = seq_length
        self.set_stride = set_stride
        self.instance_size = instance_size
        self.train_paths, self.val_paths = self.get_training_paths()

    def get_training_paths(self):
        return get_training_station_paths(self.data_path)

    def get_path_labels(self, paths):
        labels = [path.split('/')[-1] for path in paths]
        return [self.get_label_one_hot(label) for label in labels]

    # ------------------- Multi-Input -------------------

    def load_masked_image_multi_input(self, image_path):
        with h5py.File(image_path, 'r') as file:
            image = file['image'][:]
            mask = file['mask'][:]

            image = np.repeat(image, 3, axis=-1) # repeat grayscale image to create 3 channels
            return image, mask

    def load_image_sequence_multi_input(self, frame_paths):
        images_sequence = []
        masks_sequence = []
        for frame_path in frame_paths:
            image, mask = self.load_masked_image_multi_input(frame_path)
            images_sequence.append(image)
            masks_sequence.append(mask)

        # Ensure sequences are padded to the desired length
        while len(images_sequence) < self.seq_length:
            images_sequence.append(np.zeros((self.instance_size[0], self.instance_size[1], 3)))
            masks_sequence.append(np.zeros((self.instance_size[0], self.instance_size[1], 3)))

        images_sequence = tf.stack(images_sequence)
        masks_sequence = tf.stack(masks_sequence)
        return images_sequence, masks_sequence

    # Function to create image sequences and mask sequences of seq_length for one station for one patient
    def create_multi_input_sequence(self, station_path):
        station_path = station_path.numpy().decode('utf-8')  # converts station path from tf tensor to string
        frame_paths = get_frame_paths(station_path, self.model_type)
        num_frames = len(frame_paths)

        if self.set_stride: stride = random.randint(1, 3)
        else: stride = 1

        if num_frames < (self.seq_length * stride): start_index = 0
        else: start_index = random.randint(0, num_frames - (self.seq_length * stride))

        end_index = start_index + (self.seq_length * stride)
        sequence_paths = frame_paths[start_index:end_index:stride]  # len(sequence_paths) = seq_length (or less)

        image_sequence, mask_sequence = self.load_image_sequence_multi_input(sequence_paths)

        # Return a tuple of images and masks
        return image_sequence, mask_sequence

    def data_augmentationSeq(self, x):
        augmentation_layers_segmentation = [GammaTransform(low=0.5, high=1.5), Blur(sigma_max=1.0),
                                            ContrastScale(min_scale=0.5, max_scale=1.5),
                                            GaussianShadow(sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.9), strength=(0.5, 0.8))]

        return RandomAugmentationSequence(augmentation_layers_segmentation)(x)

    @tf.function
    def data_augmentation_map_multi_input(self, x, y):
        augmented_x = tf.py_function(func=self.data_augmentationSeq, inp=[x[0]], Tout=tf.float32)
        return (augmented_x, x[1]), y


    def loader_function_multi_input(self):

        train_ds = tf.data.Dataset.from_tensor_slices((self.train_paths, self.get_path_labels(self.train_paths)))
        val_ds = tf.data.Dataset.from_tensor_slices((self.val_paths, self.get_path_labels(self.val_paths)))

        # shuffle train_ds
        train_ds = train_ds.shuffle(buffer_size=len(self.train_paths), reshuffle_each_iteration=True)

        # creates sequence of image paths for each station using tf.py_function
        train_ds = train_ds.map(lambda x, y: (tf.py_function(func=self.create_multi_input_sequence,
                                                                inp=[x], Tout=[tf.float32, tf.float32]), y),
                                    num_parallel_calls=tf.data.AUTOTUNE)

        val_ds = val_ds.map(lambda x, y: (tf.py_function(func=self.create_multi_input_sequence,
                                                            inp=[x], Tout=[tf.float32, tf.float32]), y),
                                num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment:
            # apply data augmentation to training data
            train_ds = train_ds.map(self.data_augmentation_map_multi_input, num_parallel_calls=tf.data.AUTOTUNE)

        if self.model_arch in ['resnet-lstm', 'mobileNetV3Small-lstm']:
            # scale each image and mask back to [0, 255]
            train_ds = train_ds.map(lambda x, y: ((x[0] * 255.0, x[1] * 255.0), y), num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(lambda x, y: ((x[0] * 255.0, x[1] * 255.0), y), num_parallel_calls=tf.data.AUTOTUNE)

        if self.model_arch in ['inception-lstm', 'mobilenetV2-lstm']:
            # scale only images from [0, 1] to [-1, 1]
            train_ds = train_ds.map(lambda x, y: ((x[0] - 0.5) * 2.0, x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(lambda x, y: ((x[0] - 0.5) * 2.0, x[1], y), num_parallel_calls=tf.data.AUTOTUNE)


        self.train_ds = train_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = val_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return self.train_ds, self.val_ds


# not used TODO: remove ?
# ------------------- SINGLE INPUT ------------------------

    def load_masked_image(self, image_path):
        with h5py.File(image_path, 'r') as file:
            image = file['image'][:]
            mask = file['mask'][:]

            # concatenate image and last two channels of mask
            masked_image = np.concatenate((image, mask[:, :, 1:]), axis=-1)
            return masked_image

    def load_image_sequence(self, frame_paths):
        masked_images_sequence = []
        for frame_path in frame_paths:
            masked_image = self.load_masked_image(frame_path)
            masked_images_sequence.append(masked_image)


        # Ensure sequences are padded to the desired length
        while len(masked_images_sequence) < self.seq_length:
            masked_images_sequence.append(np.zeros_like(masked_images_sequence[0]))


        masked_images_sequence = np.stack(masked_images_sequence)
        return masked_images_sequence

    # Function to create image sequences and mask sequences of seq_length for one station for one patient
    def create_sequence(self, station_path):
        frame_paths = get_frame_paths(station_path, self.model_type)
        num_frames = len(frame_paths)

        if self.set_stride: stride = random.randint(1, 3)
        else: stride = 1

        if num_frames < (self.seq_length * stride): start_index = 0
        else: start_index = random.randint(0, num_frames - (self.seq_length * stride))

        end_index = start_index + (self.seq_length * stride)
        sequence_paths = frame_paths[start_index:end_index:stride]  # len(sequence_paths) = seq_length (or less)

        masked_images_sequence = self.load_image_sequence(sequence_paths)

        return masked_images_sequence

    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
    def gen(self, paths):
        while True:  # infinite generator
            #choose a random station from the list of stations
            station = random.choice(list(self.stations_config.keys()))
            # choose a random patient from the list of patients for that station
            filtered_paths = [path for path in paths if path.endswith(station)]
            patient = random.choice(filtered_paths)
            yield self.create_sequence(patient), self.get_label_one_hot(station)


    def loader_function(self):
        # create dataset from generator (added lambda to provide a callable function to .from_generator)
        train_ds = tf.data.Dataset.from_generator(lambda: self.gen(self.train_paths),
                                                  output_types=(tf.float32, tf.float32),
                                                  output_shapes=(tf.TensorShape([None, None, None, 3]), tf.TensorShape([self.num_stations])))

        val_ds = tf.data.Dataset.from_generator(lambda: self.gen(self.val_paths),
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=(tf.TensorShape([None, None, None, 3]), tf.TensorShape([self.num_stations])))


        if self.model_arch == 'mobileNetV3Small-lstm':
            # scale each image back to [0, 255]
            train_ds = train_ds.map(lambda x, y: (x * 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(lambda x, y: (x * 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)

        self.train_ds = train_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = val_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return self.train_ds, self.val_ds
