import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src.models.training.experimentLogger import ExperimentLogger
from mlmia.training import enable_gpu_growth
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from src.resources.loss import get_loss
from src.resources.train_config import set_train_config
from src.data.classification_pipeline import EBUSClassificationPipeline
from src.resources.ml_models import get_arch
from src.utils.get_class_weight import get_class_weight

enable_gpu_growth()
logger = logging.getLogger()

def train_model(data_path, log_path, image_shape, validation_split, test_split,
                batch_size, stations_config, num_stations, loss, model_arch,
                instance_size, learning_rate, model_path, patience,
                epochs, augment, stratified_cv):

    print("Trainer: " + model_arch)
    trainer = BaselineTrainer(data_path, log_path, image_shape, validation_split, test_split, batch_size,
                              stations_config, num_stations, loss, model_arch, instance_size,
                              learning_rate, model_path, patience, epochs, augment, stratified_cv)
    # Perform training
    trainer.train()

    return trainer

class BaselineTrainer:

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
        self.log_path = log_path
        self.image_shape = image_shape
        self.validation_split = validation_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.stations_config = stations_config
        self.num_stations = num_stations
        self.loss = loss
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

    #-----------------------------  PREPROCESSING ----------------------------------

    # Set up pipeline
    def preprocess(self):
        self.pipeline = EBUSClassificationPipeline(data_path=self.data_path,
                                                   batch_size=self.batch_size,
                                                   image_shape=self.image_shape,
                                                   validation_split=self.validation_split,
                                                   station_names=list(self.stations_config.keys()),
                                                   num_stations=self.num_stations,
                                                   augment=self.augment
                                                   )

        self.train_ds, self.val_ds = self.pipeline.loader_function()

        # MULTICLASS : images: shape=(32, 256, 256, 3), labels: shape=(32, 9) for batch_size=32
        # BINARY : labels: tf.Tensor([0 1 0 1 1 0 1 1 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 1 0], shape=(32,), dtype=int32)
        plt.figure(figsize=(10, 10))
        class_names = list(self.stations_config.keys())
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i])
                plt.axis("off")
                if self.num_stations > 2: plt.title(class_names[np.argmax(labels[i])])
                else: plt.title(class_names[labels.numpy()[i]]) #to get class label from tf.Tensor(0, shape=(), dtype=int32)
        plt.show()


    # -----------------------------  BUILDING AND SAVING MODEL ----------------------------------

    def build_model(self):

        self.model = get_arch(self.model_arch, self.instance_size, self.num_stations)
        print(self.model.summary())

        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss=get_loss(self.loss),
            metrics=['accuracy', Precision(), Recall()]
        )

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)

        # make experiment logger
        train_config = set_train_config()
        self.experiment_logger = ExperimentLogger(logdir=train_config.log_directory,
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
            self.train_stratified_cv() # Allocation of 10934550528 exceeds 10% of free system memory. Might be useful for small dataset

        else:
            save_best, early_stop = self.save_model()

            self.model.fit(self.train_ds,
                           epochs=self.epochs,
                           validation_data=self.val_ds,
                           callbacks=[save_best, early_stop, self.experiment_logger])
                           #class_weight=get_class_weight(self.train_ds, self.num_stations))

            best_model = tf.keras.models.load_model(self.experiment_logger.get_latest_checkpoint(), compile=False)
            best_model.save(os.path.join(str(self.experiment_logger.logdir), 'best_model'))

