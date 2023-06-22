import os
import logging

from mlmia.logger import ExperimentLogger
from mlmia.training import enable_gpu_growth
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from src.resources.loss_dict import get_loss
from src.resources.train_config import set_train_config
from src.data.classification_pipeline import EBUSClassificationPipeline
from src.resources.ml_models import get_arch
import tensorflow as tf


enable_gpu_growth()
logger = logging.getLogger()

def train_model(stratified_cv, data_path, log_path, image_shape, tf_dataset, validation_split, test_split,
                batch_size, split_by, station_config_nr, loss, augment_data, model_arch,
                instance_size, learning_rate, model_path, history_path, model_name, patience,
                epochs):

    print("Trainer: " + model_arch)
    trainer = BaselineTrainer(data_path, log_path, image_shape, tf_dataset, validation_split, test_split, batch_size,
                                split_by, station_config_nr, loss, augment_data, model_arch,
                                instance_size, learning_rate, model_path, history_path,
                                model_name, patience, epochs)

    # Perform training
    trainer.train(stratified_cv)

class BaselineTrainer:

    def __init__(self,
                 data_path: str,
                 log_path: str,
                 image_shape: tuple,
                 tf_dataset: bool,
                 validation_split: float,
                 test_split: float,
                 batch_size: int,
                 split_by: str,
                 station_config_nr: int,
                 loss: str,
                 augment_data: bool,
                 model_arch: str,
                 instance_size: tuple,
                 learning_rate: float,
                 model_path: str,
                 history_path: str,
                 model_name: str,
                 patience: int,
                 epochs: int
                 ):
        self.data_path = data_path
        self.log_path = log_path
        self.image_shape = image_shape
        self.tf_dataset = tf_dataset
        self.validation_split = validation_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.split_by = split_by
        self.station_config_nr = station_config_nr
        self.loss = loss
        self.augment_data = augment_data
        self.model_arch = model_arch
        self.instance_size = instance_size
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.history_path = history_path
        self.model_name = model_name
        self.patience = patience
        self.epochs = epochs


        self.pipeline = None
        self.model = None
        self.generator = None
        self.callbacks = None

    #-----------------------------  PREPROCESSING ----------------------------------

    # Set up pipeline
    def preprocess(self):
        self.pipeline = EBUSClassificationPipeline(data_path=self.data_path,
                                   image_shape=self.image_shape,
                                   tf_dataset=self.tf_dataset,
                                   validation_split=self.validation_split,
                                   test_split=self.test_split,
                                   batch_size=self.batch_size,
                                   split_by=self.split_by,
                                   station_config_nr=self.station_config_nr,
                                   )
        if self.augment_data:
            self.pipeline.data_augmentor.add_rotation(max_angle=30, apply_to=(0,))
            self.pipeline.data_augmentor.add_gamma_transformation(0.5, 1.5)
            self.pipeline.data_augmentor.add_gaussian_shadow()

        self.generator = self.pipeline.generator_containers[0]

        print('Training subjects', self.generator.training.get_subjects())
        print('Validation subjects', self.generator.validation.get_subjects())

    # -----------------------------  BUILDING AND SAVING MODEL ----------------------------------

    def build_model(self):
        self.model = get_arch(self.model_arch, self.instance_size, self.pipeline.get_num_stations())
        print(self.model.summary())

        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss=get_loss(self.loss),
            metrics=['accuracy', Precision(), Recall()] # TODO: add precision and recall, without error
        )

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.history_path, exist_ok=True)

        # make experiment logger
        self.train_config = set_train_config()
        self.experiment_logger = ExperimentLogger(logdir=self.train_config.log_directory,
                                                  pipeline_config=self.pipeline.get_config(),
                                                  train_config=self.train_config.get_config())
        self.experiment_logger.save_current_config()

        save_best = ModelCheckpoint(filepath=self.experiment_logger.create_checkpoint_filepath(),
                                    monitor='val_loss',
                                    verbose=1, save_best_only=True,
                                    save_weights_only=False)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                   patience=self.train_config.early_stop_patience, verbose=1)

        return save_best, early_stop

    # -----------------------------  TRAINING ----------------------------------

    def train_stratified_cv(self):
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        print(self.generator.load_data())



    def train(self, stratified_cv: bool):
        self.preprocess()
        self.build_model()

        print("-- TRAINING --")
        if stratified_cv:
            self.train_stratified_cv()  # Train model with stratified cross validation

        else:
            save_best, early_stop = self.save_model()

            self.model.fit(self.generator.training,
                      epochs=self.epochs,
                      steps_per_epoch=self.generator.training.steps_per_epoch,
                      validation_data=self.generator.validation,
                      validation_steps=self.generator.validation.steps_per_epoch,
                      callbacks=[save_best, early_stop, self.experiment_logger])

            best_model = tf.keras.models.load_model(self.experiment_logger.get_latest_checkpoint(), compile=False)
            best_model.save(os.path.join(str(self.experiment_logger.logdir), 'best_model'))
