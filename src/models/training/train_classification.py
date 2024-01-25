import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import F1Score

from src.models.training.experimentLogger import ExperimentLogger
from mlmia.training import enable_gpu_growth
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from contextlib import redirect_stdout
from src.resources.loss import get_loss
from src.resources.train_config import set_train_config
from src.data.classification_pipeline import BaselineClassificationPipeline, SequenceClassificationPipeline
from src.resources.ml_models import get_arch
from src.utils.get_class_weight import get_class_weight
from src.utils.metric_callback import ClassMetricsCallback
from src.utils.timing_callback import TimingCallback

enable_gpu_growth()
logger = logging.getLogger()

def train_model(data_path, test_ds_path, log_path, image_shape, validation_split, test_split,
                batch_size, stations_config, num_stations, loss, model_type, model_arch,
                instance_size, learning_rate, model_path, patience,
                epochs, steps_per_epoch, validation_steps, stride, augment, stratified_cv, seq_length):

    if model_type == "baseline" or model_type == "combined_baseline":
        trainer = BaselineTrainer(data_path, test_ds_path, log_path, image_shape, validation_split, test_split,
                                  batch_size, stations_config, num_stations, loss, model_type, model_arch,
                                  instance_size, learning_rate, model_path, patience, epochs, augment, stratified_cv)
    elif model_type == "sequence" or model_type == "combined_sequence":
        trainer = SequenceTrainer(data_path, test_ds_path, log_path, image_shape, validation_split, test_split,
                                  batch_size, stations_config, num_stations, loss, model_type, model_arch,
                                  instance_size, learning_rate, model_path, patience, epochs, steps_per_epoch,
                                  validation_steps, stride, augment, seq_length)
    else:
        raise ValueError("Model type not supported")

    # Perform training
    trainer.train()

    return trainer


class BaselineTrainer:

    def __init__(self,
                 data_path: str,
                 test_ds_path: str,
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
                 stratified_cv: bool
                 ):
        self.data_path = data_path
        self.test_ds_path = test_ds_path
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
        self.stratified_cv = stratified_cv

        self.pipeline = None
        self.model = None
        self.callbacks = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    # -----------------------------  PREPROCESSING ----------------------------------

    # Set up pipeline
    def preprocess(self):
        self.pipeline = BaselineClassificationPipeline(data_path=self.data_path,
                                                       test_ds_path=self.test_ds_path,
                                                       batch_size=self.batch_size,
                                                       image_shape=self.image_shape,
                                                       validation_split=self.validation_split,
                                                       test_split=self.test_split,
                                                       station_names=list(self.stations_config.keys()),
                                                       num_stations=self.num_stations,
                                                       augment=self.augment
                                                       )
        self.train_ds, self.val_ds, self.test_ds = self.pipeline.loader_function()

        # MULTICLASS : images: shape=(32, 256, 256, 3), labels: shape=(32, 9) for batch_size=32
        # BINARY : labels: tf.Tensor([0 1 0 1 1 0 1 1 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 1 0], shape=(32,), dtype=int32)
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                # normalize image from range [-1, 1] to [0, 1]
                image = (images[i] + 1) / 2
                #image = images[i] / 255
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
    # Not used at the moment, TODO: remove?
    def train_stratified_cv(self):
        # Stratified K-fold cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        train_images = np.concatenate(list(self.train_ds.map(lambda x, y: x)))
        train_labels = np.concatenate(list(self.train_ds.map(lambda x, y: y)))
        train_labels = np.argmax(train_labels, axis=1)

        val_images = np.concatenate(list(self.val_ds.map(lambda x, y: x)))
        val_labels = np.concatenate(list(self.val_ds.map(lambda x, y: y)))
        val_labels = np.argmax(val_labels, axis=1)

        images = np.concatenate((train_images, val_images), axis=0)
        labels = np.concatenate((train_labels, val_labels), axis=0)

        # Iterate through folds
        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            print(f"Fold: {fold + 1}")

            train_dataset = tf.data.Dataset.from_tensor_slices((images[train_idx], labels[train_idx]))
            val_dataset = tf.data.Dataset.from_tensor_slices((images[val_idx], labels[val_idx]))

            self.train_ds = train_dataset.shuffle(1024).batch(self.batch_size)
            self.val_ds = val_dataset.batch(self.batch_size)

            # Save best model and early stop
            save_best, early_stop = self.save_model()

            # Train model
            self.model.fit(self.train_ds,
                           epochs=self.epochs,
                           validation_data=self.val_ds,
                           callbacks=[save_best, early_stop, self.experiment_logger],
                           class_weight=get_class_weight(self.train_ds, self.num_stations))

            # Load best model and save it
            best_model = tf.keras.models.load_model(self.experiment_logger.get_latest_checkpoint(), compile=False)
            best_model.save(os.path.join(str(self.experiment_logger.logdir), f'best_model_{fold}'))

    def train(self):
        self.preprocess()
        self.build_model()

        print("-- TRAINING --")
        # Train model with stratified cross validation
        if self.stratified_cv:
            self.train_stratified_cv()  # Allocation of 10934550528 exceeds 10% of free system memory. Might be useful for small dataset

        else:
            save_best, early_stop = self.save_model()

            self.model.fit(self.train_ds,
                           epochs=self.epochs,
                           validation_data=self.val_ds,
                           callbacks=[save_best, early_stop, self.experiment_logger],
                           class_weight=get_class_weight(self.train_ds, self.num_stations))

            best_model = tf.keras.models.load_model(self.experiment_logger.get_latest_checkpoint(), compile=False)
            best_model.save(os.path.join(str(self.experiment_logger.logdir), 'best_model'))


class SequenceTrainer:

    def __init__(self,
                 data_path: str,
                 test_ds_path: str,
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
                 steps_per_epoch: int,
                 validation_steps: int,
                 stride: int,
                 augment: bool,
                 seq_length: int
                 ):
        self.data_path = data_path
        self.test_ds_path = test_ds_path
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
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.stride = stride
        self.augment = augment
        self.seq_length = seq_length

        self.pipeline = None
        self.model = None
        self.callbacks = None
        self.train_ds = None
        self.val_ds = None

    # -----------------------------  PREPROCESSING ----------------------------------

    # Set up pipeline
    def preprocess(self):

        self.pipeline = SequenceClassificationPipeline(data_path=self.data_path,
                                                       test_ds_path=self.test_ds_path,
                                                       batch_size=self.batch_size,
                                                       image_shape=self.image_shape,
                                                       validation_split=self.validation_split,
                                                       test_split=self.test_split,
                                                       station_names=list(self.stations_config.keys()),
                                                       num_stations=self.num_stations,
                                                       augment=self.augment,
                                                       stations_config=self.stations_config,
                                                       seq_length=self.seq_length,
                                                       stride=self.stride
                                                       )

        self.train_ds, self.val_ds = self.pipeline.loader_function()

        plt.style.use('dark_background')
        # print the value range of the pixels
        for images, labels in self.train_ds.take(1):
            print('images shape: ', images.shape)  # (batch_size, seq_length, 256, 256, 3)
            print(tf.reduce_min(images), tf.reduce_max(images))

            for i in range(self.batch_size):
                plt.imshow(images[i][0] / 255)
                plt.title(f'Label: {labels[i]} (batch {i})')
                plt.axis('off')
                plt.show()


        '''
        # plotting the seq_length first frames of the first sequence in the first four batches
        for i, (images, labels) in enumerate(self.train_ds.take(4)):
            print('images shape: ', images.shape)  # (4, seq_length, 256, 256, 3)
            print('labels shape: ', labels.shape)  # (4, 8)
            rows = self.seq_length // 2 if self.seq_length % 2 == 0 else self.seq_length // 2 + 1

            #plt.style.use('ggplot')
            plt.figure(figsize=(10, 10))
            for j in range(self.seq_length):
                plt.subplot(rows, 3, j + 1)
                # normalize image from range [-1, 1] to [0, 255]
                # image = (images[0][j] + 1) * 127.5
                # image = images[i][j] * 255  # for range [0, 1]
                plt.imshow(np.array(images[i][j]).astype("uint8"))
                plt.title(f"Frame {j}, Label: {self.pipeline.station_names[np.argmax(labels.numpy()[0])]}", fontsize=10)
                plt.axis("off")
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Adjust top spacing for suptitle
            plt.suptitle(f"Batch {i}")
            plt.show()
            
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
        print("Number of training sequences: ", self.steps_per_epoch)
        print("Number of validation sequences: ", self.validation_steps)

        save_best, early_stop, tb_logger, class_metrics = self.save_model()

        self.model.fit(self.train_ds,
                       epochs=self.epochs,
                       validation_data=self.val_ds,
                       steps_per_epoch=self.steps_per_epoch, # number of sequences / batch_size
                       validation_steps=self.validation_steps,
                       callbacks=[save_best, early_stop, self.experiment_logger, tb_logger, class_metrics])
                       #class_weight=get_class_weight(self.train_ds.take(self.steps_per_epoch), self.num_stations))

        best_model = tf.keras.models.load_model(self.experiment_logger.get_latest_checkpoint(), compile=False)
        best_model.save(os.path.join(str(self.experiment_logger.logdir), 'best_model'))
