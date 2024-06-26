import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow_addons.metrics import F1Score
from matplotlib.colors import LinearSegmentedColormap
from src.models.training.experimentLogger import ExperimentLogger
from mlmia.training import enable_gpu_growth
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from contextlib import redirect_stdout
from src.resources.loss import get_loss
from src.resources.train_config import set_train_config
from src.data.classification_pipeline import Baseline, Sequence, SequenceWithSegmentation
from src.resources.architectures.ml_models import get_arch
from src.utils.get_class_weight import get_class_weight
from src.utils.metric_callback import ClassMetricsCallback

enable_gpu_growth()
logger = logging.getLogger()

def train_model(data_path, log_path, image_shape, validation_split, test_split,
                batch_size, stations_config, num_stations, loss, model_type, model_arch,
                instance_size, learning_rate, model_path, patience,
                epochs, steps_per_epoch, validation_steps, set_stride, augment, stratified_cv, seq_length, full_video, use_quality_weights, use_gen):

    if model_type == "baseline":
        trainer = BaselineTrainer(data_path, log_path, image_shape, validation_split, test_split,
                                  batch_size, stations_config, num_stations, loss, model_type, model_arch,
                                  instance_size, learning_rate, model_path, patience, epochs, augment, stratified_cv, use_quality_weights)
    elif model_type == "sequence" or model_type == "sequence_cv":
        trainer = SequenceTrainer(data_path, log_path, image_shape, validation_split, test_split,
                                  batch_size, stations_config, num_stations, loss, model_type, model_arch,
                                  instance_size, learning_rate, model_path, patience, epochs, steps_per_epoch,
                                  validation_steps, set_stride, augment, seq_length, full_video, use_gen, use_quality_weights)
    elif model_type == "sequence_with_segmentation":
        trainer = SequenceWithSegmentationTrainer(data_path, log_path, image_shape, validation_split, test_split,
                                  batch_size, stations_config, num_stations, loss, model_type, model_arch,
                                  instance_size, learning_rate, model_path, patience, epochs, steps_per_epoch,
                                  validation_steps, set_stride, augment, seq_length)
    else:
        raise ValueError("Model type not supported")

    # Perform training
    print('data_path: ', data_path)
    trainer.train()

    return trainer

class Trainer:
    def __init__(self,
                 data_path: str,
                 log_path: str,
                 image_shape: tuple,
                 validation_split: float,
                 test_split: float,
                 batch_size: int,
                 stations_config: dict,
                 num_stations: int,
                 loss: str,
                 model_type: str,
                 model_arch: str,
                 instance_size: tuple,
                 learning_rate: float,
                 model_path: str,
                 patience: int,
                 epochs: int,
                 augment: bool,
                 ):

        self.data_path = data_path
        self.log_path = log_path
        self.image_shape = image_shape
        self.validation_split = validation_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.stations_config = stations_config
        self.num_stations = num_stations
        self.loss = loss
        self.model_type = model_type
        self.model_arch = model_arch
        self.instance_size = instance_size
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.patience = patience
        self.epochs = epochs
        self.augment = augment

        self.pipeline = None
        self.model = None
        self.callbacks = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None



class BaselineTrainer(Trainer):

    def __init__(self, data_path, log_path, image_shape, validation_split, test_split, batch_size, stations_config,
                 num_stations, loss, model_type, model_arch, instance_size, learning_rate, model_path, patience, epochs,
                 augment, stratified_cv, use_quality_weights):
        super().__init__(data_path, log_path, image_shape, validation_split, test_split, batch_size, stations_config,
                         num_stations, loss, model_type, model_arch, instance_size, learning_rate, model_path, patience,
                         epochs, augment)

        self.use_quality_weights = use_quality_weights

    # -----------------------------  PREPROCESSING ----------------------------------

    # Set up pipeline
    def preprocess(self):
        self.pipeline = Baseline(data_path=self.data_path,
                                 batch_size=self.batch_size,
                                 image_shape=self.image_shape,
                                 validation_split=self.validation_split,
                                 test_split=self.test_split,
                                 station_names=list(self.stations_config.keys()),
                                 num_stations=self.num_stations,
                                 augment=self.augment,
                                 stations_config=self.stations_config,
                                 model_arch=self.model_arch,
                                 )
        if self.use_quality_weights:
            self.train_ds, self.val_ds, self.test_ds = self.pipeline.loader_function_with_quality()

            # print the shape of the dataset
            for images, labels, quality in self.train_ds.take(1):
                print(images.shape, labels.shape, quality.shape)

            plt.style.use('ggplot')
            plt.figure(figsize=(10, 10))
            for images, labels, quality in self.train_ds.take(1):
                for i in range(9):
                    ax = plt.subplot(3, 3, i + 1)
                    # normalize image from range [-1, 1] to [0, 255]
                    image = (images[i] + 1. / 2.)
                    #image = images[i] / 255
                    plt.imshow(image)
                    plt.axis("off")
                    if self.num_stations > 2:
                        plt.title(self.pipeline.station_names[np.argmax(labels[i])] + ' train' + ' quality: ' + str(quality[i].numpy()))
                    else:
                        plt.title(
                            self.pipeline.station_names[
                                labels.numpy()[i]])  # to get class label from tf.Tensor(0, shape=(), dtype=int32)
            plt.show()

        else:
            self.train_ds, self.val_ds, self.test_ds = self.pipeline.loader_function()

            # MULTICLASS : images: shape=(32, 256, 256, 3), labels: shape=(32, 9) for batch_size=32
            # BINARY : labels: tf.Tensor([0 1 0 1 1 0 1 1 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 1 0], shape=(32,), dtype=int32)
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 10))
            for images, labels in self.train_ds.take(1):
                for i in range(9):
                    ax = plt.subplot(3, 3, i + 1)
                    # normalize image from range [-1, 1] to [0, 1]
                    #image = (images[i] + 1) / 2
                    image = images[i] / 255
                    plt.imshow(image)
                    plt.axis("off")
                    if self.num_stations > 2:
                        plt.title(self.pipeline.station_names[np.argmax(labels[i])] + ' train')
                    else:
                        plt.title(
                            self.pipeline.station_names[
                                labels.numpy()[i]])  # to get class label from tf.Tensor(0, shape=(), dtype=int32)
            plt.show()
    # -----------------------------  BUILDING AND SAVING MODEL ----------------------------------

    def build_model(self):

        self.model = get_arch(self.model_arch, self.instance_size, self.num_stations)
        print(self.model.summary())
        if self.use_quality_weights:
            self.model.compile(
                optimizer=Adam(self.learning_rate),
                loss=get_loss(self.loss),
                weighted_metrics=['accuracy', Precision(), Recall()]
            )
        else:
            self.model.compile(
                optimizer=Adam(self.learning_rate),
                loss=get_loss(self.loss),
                metrics=['accuracy', Precision(), Recall(), F1Score(self.num_stations, average='macro')]
            )

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)

        # save self.model.summary()
        with open(os.path.join(self.model_path, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        # make experiment logger
        train_config = set_train_config()
        self.experiment_logger = ExperimentLogger(logdir=train_config.model_directory,
                                                  train_config=train_config.get_config())
        self.experiment_logger.save_current_config()

        save_best = ModelCheckpoint(filepath=self.experiment_logger.create_checkpoint_filepath(),
                                    monitor='val_loss',
                                    verbose=1, save_best_only=True,
                                    save_weights_only=False)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                   patience=train_config.early_stop_patience, verbose=1)

        return save_best, early_stop

    # -----------------------------  TRAINING ----------------------------------
    def train(self):
        self.preprocess()
        self.build_model()

        print("-- TRAINING --")

        save_best, early_stop = self.save_model()

        self.model.fit(self.train_ds,
                       epochs=self.epochs,
                       validation_data=self.val_ds,
                       callbacks=[save_best, early_stop, self.experiment_logger])
                       #class_weight=get_class_weight(self.train_ds, self.num_stations))

        best_model = tf.keras.models.load_model(self.experiment_logger.get_latest_checkpoint(), compile=False)
        best_model.save(os.path.join(str(self.experiment_logger.logdir), 'best_model'))


class SequenceTrainer(Trainer):
    def __init__(self, data_path, log_path, image_shape, validation_split, test_split,
                 batch_size, stations_config, num_stations, loss, model_type, model_arch,
                 instance_size, learning_rate, model_path, patience, epochs, steps_per_epoch,
                 validation_steps, set_stride, augment, seq_length, full_video, use_gen, use_quality_weights):
        super().__init__(data_path, log_path, image_shape, validation_split, test_split, batch_size, stations_config,
                         num_stations, loss, model_type, model_arch, instance_size, learning_rate, model_path, patience,
                         epochs, augment)

        self.seq_length = seq_length
        self.set_stride = set_stride
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.full_video = full_video
        self.use_gen = use_gen
        self.use_quality_weights = use_quality_weights



    # -----------------------------  PREPROCESSING ----------------------------------


    # Set up pipeline
    def preprocess(self):

        self.pipeline = Sequence(data_path=self.data_path,
                                 model_type=self.model_type,
                                 batch_size=self.batch_size,
                                 image_shape=self.image_shape,
                                 validation_split=self.validation_split,
                                 test_split=self.test_split,
                                 station_names=list(self.stations_config.keys()),
                                 num_stations=self.num_stations,
                                 augment=self.augment,
                                 stations_config=self.stations_config,
                                 seq_length=self.seq_length,
                                 set_stride=self.set_stride,
                                 model_arch=self.model_arch,
                                 instance_size=self.instance_size,
                                 full_video=self.full_video,
                                 use_gen=self.use_gen
                                 )

        if self.use_quality_weights:
            self.train_ds, self.val_ds = self.pipeline.loader_function_with_quality()

            plt.style.use('ggplot')
            # print the value range of the pixels
            for images, labels, qualities in self.train_ds.take(1):
                print('images shape: ', images.shape)  # (batch_size, seq_length, 256, 256, 3)
                print('min:', tf.reduce_min(images))
                print('max:', tf.reduce_max(images))
            num_images = 0

            for i, (images, labels, qualities) in enumerate(self.train_ds.take(3)):
                print('images shape: ', images.shape)  # (4, seq_length, 256, 256, 3)
                print('labels shape: ', labels.shape)  # (4, 8)
                print('qualities shape: ', qualities.shape)  # (4, seq_length, 1)
                for seq in range(images.shape[0]):
                    plt.figure(figsize=(10, 10))
                    for image in range(self.seq_length):
                        num_images += 1
                        plt.subplot(4, 4, image + 1)
                        #normalize from [-1,1] to [0, 255]
                        image_norm = (images[seq][image] + 1. / 2.) * 255
                        plt.imshow(np.array(image_norm).astype("uint8"))
                        plt.title(
                            f"Frame {image}, Label: {self.pipeline.station_names[np.argmax(labels.numpy()[seq])]}",
                            fontsize=10)
                        plt.axis("off")
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.85)
                    plt.suptitle(f"Batch {i}, Sequence {seq}, Quality score {qualities.numpy()[seq]}", fontsize=16)
                    plt.show()
            print(f"Total images: {num_images}")

        else:
            self.train_ds, self.val_ds = self.pipeline.loader_function()
            '''
            plt.style.use('ggplot')
            # print the value range of the pixels
            for images, labels in self.train_ds.take(1):
                print('images shape: ', images.shape)  # (batch_size, seq_length, 256, 256, 3)
                print('min:', tf.reduce_min(images))
                print('max:', tf.reduce_max(images))
            num_images = 0

            for i, (images, labels) in enumerate(self.train_ds.take(3)):
                print('images shape: ', images.shape)  # (4, seq_length, 256, 256, 3)
                print('labels shape: ', labels.shape)  # (4, 8)
                for seq in range(images.shape[0]):
                    plt.figure(figsize=(10, 10))
                    for image in range(self.seq_length):
                        num_images += 1
                        plt.subplot(4, 4, image + 1)
                        image_norm = (images[seq][image] + 1) * 127.5
                        plt.imshow(np.array(image_norm).astype("uint8"))
                        plt.title(f"Frame {image}, Label: {self.pipeline.station_names[np.argmax(labels.numpy()[seq])]}",
                                  fontsize=10)
                        plt.axis("off")
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.85)
                    plt.suptitle(f"Batch {i}, Sequence {seq}", fontsize=16)
                    plt.show()
            print(f"Total images: {num_images}")
            '''

    # -----------------------------  BUILDING AND SAVING MODEL ----------------------------------

    def build_model(self):
        print('num_stations: ', self.num_stations)

        self.model = get_arch(self.model_arch, self.instance_size, self.num_stations, stateful=False)
        print(self.model.summary())

        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss=get_loss(self.loss),
            metrics=['accuracy', Precision(), Recall(), F1Score(self.num_stations, average='macro')]
            # macro treats all classes equally
        )

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)

        # save self.model.summary()
        with open(os.path.join(self.model_path, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        # make experiment logger
        train_config = set_train_config()

        # make callbacks
        tb_logger = TensorBoard(log_dir=os.path.join('logs/fit/', train_config.log_directory), histogram_freq=1,
                                update_freq="batch")

        self.experiment_logger = ExperimentLogger(logdir=self.model_path,
                                                  train_config=train_config.get_config())
        self.experiment_logger.save_current_config()

        save_best = ModelCheckpoint(filepath=self.experiment_logger.create_checkpoint_filepath(),
                                    monitor='val_loss',
                                    verbose=1, save_best_only=True,
                                    save_weights_only=False)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                   patience=train_config.early_stop_patience, verbose=1)

        # time = TimingCallback()

        class_metrics = ClassMetricsCallback(station_names=self.pipeline.station_names,
                                             train_ds=self.train_ds.take(self.steps_per_epoch),
                                             val_ds=self.val_ds.take(self.validation_steps),
                                             save_path=os.path.join(self.model_path, 'metrics.csv'))


        return save_best, early_stop, tb_logger

    # -----------------------------  TRAINING ----------------------------------

    def train(self):
        self.preprocess()
        self.build_model()

        print("-- TRAINING --")

        save_best, early_stop, tb_logger = self.save_model() #class_metrics
        if self.use_gen:
            print("Number of training sequences: ", self.steps_per_epoch)
            print("Number of validation sequences: ", self.validation_steps)
            self.model.fit(self.train_ds,
                           epochs=self.epochs,
                           validation_data=self.val_ds,
                           steps_per_epoch=self.steps_per_epoch, # number of sequences / batch_size
                           validation_steps=self.validation_steps,
                           callbacks=[save_best, early_stop, self.experiment_logger, tb_logger])
        else:
            self.model.fit(self.train_ds,
                           epochs=self.epochs,
                           validation_data=self.val_ds,
                           callbacks=[save_best, early_stop, self.experiment_logger, tb_logger])
                           #class_weight=get_class_weight(self.train_ds, self.num_stations))

        best_model = tf.keras.models.load_model(self.experiment_logger.get_latest_checkpoint(), compile=False)
        best_model.save(os.path.join(str(self.experiment_logger.logdir), 'best_model'))


class SequenceWithSegmentationTrainer(Trainer):
    def __init__(self, data_path, log_path, image_shape, validation_split, test_split, batch_size, stations_config,
                 num_stations, loss, model_type, model_arch, instance_size, learning_rate, model_path, patience, epochs,
                 steps_per_epoch, validation_steps, set_stride, augment, seq_length):
        super().__init__(data_path, log_path, image_shape, validation_split, test_split, batch_size, stations_config,
                         num_stations, loss, model_type, model_arch, instance_size, learning_rate, model_path, patience,
                         epochs, augment)

        self.seq_length = seq_length
        self.set_stride = set_stride
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps


    # -----------------------------  PREPROCESSING ----------------------------------

    # Set up pipeline
    def preprocess(self):

        self.pipeline = SequenceWithSegmentation(data_path=self.data_path,
                                                 model_type=self.model_type,
                                                 batch_size=self.batch_size,
                                                 image_shape=self.image_shape,
                                                 validation_split=self.validation_split,
                                                 test_split=self.test_split,
                                                 station_names=list(self.stations_config.keys()),
                                                 num_stations=self.num_stations,
                                                 augment=self.augment,
                                                 stations_config=self.stations_config,
                                                 seq_length=self.seq_length,
                                                 set_stride=self.set_stride,
                                                 model_arch=self.model_arch,
                                                 instance_size=self.instance_size
                                                 )


        self.train_ds, self.val_ds = self.pipeline.loader_function_multi_input()

        plt.style.use('dark_background')
        # print the value range of the pixels
        for data in self.train_ds.take(1):
            images, labels = data

            # images is first channel and masks is second and third channel
            image_input = images[0]
            mask_input = images[1]
            print('Image :', image_input.shape)  # e.g., (batch_size, seq_length, height, width, channels)
            print('Mask :', mask_input.shape)  # e.g., (batch_size, seq_length, height, width, mask_channels)

            print('image min:', tf.reduce_min(image_input))
            print('image max:', tf.reduce_max(image_input))
            print('mask min:', tf.reduce_min(mask_input))
            print('mask max:', tf.reduce_max(mask_input))


            for i in range(self.batch_size):
                c_invalid = (0, 0, 0)
                colors = [(0.2, 0.2, 0.2),  # dark gray = background
                          (0, 0.4, 0.05),  # green = lymph nodes
                          (0.76, 0.1, 0.05)]  # red   = blood vessels

                label_cmap = LinearSegmentedColormap.from_list('label_map', colors, N=3)
                label_cmap.set_bad(color=c_invalid, alpha=0)  # set invalid (nan) colors to be transparent
                image_cmap = plt.cm.get_cmap('gray')
                image_cmap.set_bad(color=c_invalid)

                image = image_input[i][0] / 2 + 0.5
                mask = mask_input[i][0]

                pred = np.argmax(mask, axis=-1)
                plt.imshow(image, cmap=image_cmap)
                plt.imshow(pred, cmap=label_cmap, interpolation='nearest', alpha=0.4, vmin=0, vmax=2)
                plt.title(f'Batch {i}, Label: {self.pipeline.station_names[np.argmax(labels.numpy()[i])]}')
                plt.axis('off')
                plt.show()

    # -----------------------------  BUILDING AND SAVING MODEL ----------------------------------

    def build_model(self):
        print('num_stations: ', self.num_stations)

        self.model = get_arch(self.model_arch, self.instance_size, self.num_stations, stateful=False)
        print(self.model.summary())

        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss=get_loss(self.loss),
            metrics=['accuracy', Precision(), Recall(), F1Score(self.num_stations, average='macro')]
        )

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)

        # save self.model.summary()
        with open(os.path.join(self.model_path, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        # make experiment logger
        train_config = set_train_config()

        # make callbacks
        tb_logger = TensorBoard(log_dir=os.path.join('logs/fit/', train_config.log_directory), histogram_freq=1,
                                update_freq="batch")

        self.experiment_logger = ExperimentLogger(logdir=train_config.model_directory,
                                                  train_config=train_config.get_config())
        self.experiment_logger.save_current_config()

        save_best = ModelCheckpoint(filepath=self.experiment_logger.create_checkpoint_filepath(),
                                    monitor='val_loss',
                                    verbose=1, save_best_only=True,
                                    save_weights_only=False)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                   patience=train_config.early_stop_patience, verbose=1)

        # time = TimingCallback()

        class_metrics = ClassMetricsCallback(station_names=self.pipeline.station_names,
                                             train_ds=self.train_ds.take(self.steps_per_epoch),
                                             val_ds=self.val_ds.take(self.validation_steps),
                                             save_path=os.path.join(self.model_path, 'metrics.csv'))

        return save_best, early_stop, tb_logger, class_metrics

    # -----------------------------  TRAINING ----------------------------------

    def train(self):
        self.preprocess()
        self.build_model()

        print("-- TRAINING --")

        save_best, early_stop, tb_logger, class_metrics = self.save_model()

        self.model.fit(self.train_ds,
                       epochs=self.epochs,
                       validation_data=self.val_ds,
                       callbacks=[save_best, early_stop, self.experiment_logger, tb_logger, class_metrics])

        best_model = tf.keras.models.load_model(self.experiment_logger.get_latest_checkpoint(), compile=False)
        best_model.save(os.path.join(str(self.experiment_logger.logdir), 'best_model'))
