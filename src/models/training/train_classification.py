import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src.models.training.experimentLogger import ExperimentLogger
from mlmia.training import enable_gpu_growth
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from src.resources.loss import get_loss
from src.resources.train_config import set_train_config
from src.data.classification_pipeline import EBUSClassificationPipeline
from src.resources.ml_models import get_arch

enable_gpu_growth()
logger = logging.getLogger()

def train_model(data_path, log_path, image_shape, validation_split, test_split,
                batch_size, stations_config, num_stations, loss, model_arch,
                instance_size, learning_rate, model_path, patience,
                epochs, augment):

    print("Trainer: " + model_arch)
    trainer = BaselineTrainer(data_path, log_path, image_shape, validation_split, test_split, batch_size,
                              stations_config, num_stations, loss, model_arch, instance_size,
                              learning_rate, model_path, patience, epochs, augment)

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
                 augment: bool
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

        plt.figure(figsize=(10, 10))
        class_names = list(self.stations_config.keys())
        for images, labels in self.train_ds.take(1): #tf.Tensor([0 1 0 1 1 0 1 1 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 1 0], shape=(32,), dtype=int32)
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i])
                plt.title(class_names[labels.numpy()[i]]) #to get class label from tf.Tensor(0, shape=(), dtype=int32)
                plt.axis("off")
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

    def train(self):
        self.preprocess()
        self.build_model()

        print("-- TRAINING --")
        save_best, early_stop = self.save_model()

        #balance data by calculating class weights and using them in fit
        #count_array = count_station_distribution(self.train_ds, self.num_stations)
        #class_weights = {idx: (1/ elem) * np.sum(count_array)/self.num_stations for idx, elem in enumerate(count_array)}
        #print(class_weights)

        self.model.fit(self.train_ds,
                       epochs=self.epochs,
                       validation_data=self.val_ds,
                       callbacks=[save_best, early_stop, self.experiment_logger])
                       #class_weight=class_weights)

        best_model = tf.keras.models.load_model(self.experiment_logger.get_latest_checkpoint(), compile=False)
        best_model.save(os.path.join(str(self.experiment_logger.logdir), 'best_model'))