import os
import logging

from mlmia.training import enable_gpu_growth
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import Precision, Recall
from src.utils.loss_dict import get_loss

from src.data.classification_pipeline import EBUSClassificationPipeline
from src.resources.ml_models import get_arch

enable_gpu_growth()
logger = logging.getLogger()

def train_model(perform_training, stratified_cv, data_path, image_shape, tf_dataset, validation_split, test_split,
                batch_size, split_by, station_config_nr, loss, augment_data, model_arch,
                instance_size, learning_rate, model_path, history_path, model_name, patience,
                epochs):

    print("Trainer: " + model_arch)
    trainer = BaselineTrainer(perform_training, data_path, image_shape, tf_dataset, validation_split, test_split, batch_size,
                                split_by, station_config_nr, loss, augment_data, model_arch,
                                instance_size, learning_rate, model_path, history_path,
                                model_name, patience, epochs)

    # Perform training
    trainer.train(perform_training, stratified_cv)
    return trainer

class BaselineTrainer:

    def __init__(self,
                 perform_training: bool,
                 data_path: str,
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
        self.perform_training = perform_training
        self.data_path = data_path
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
        if self.perform_training:
            print('Training subjects', self.generator.training.get_subjects())
            print('Validation subjects', self.generator.validation.get_subjects())

    # -----------------------------  BUILDING AND SAVING MODEL ----------------------------------

    def build_model(self):
        self.model = get_arch(self.model_arch, self.instance_size, self.pipeline.get_num_stations())
        if self.perform_training:
            print(self.model.summary())

        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss=get_loss(self.loss),
            metrics=['accuracy', Precision(), Recall()],
        )

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.history_path, exist_ok=True)

        model_checkpoint_path = os.path.join(self.model_path, self.model_name)

        save_best = ModelCheckpoint(filepath=model_checkpoint_path,
                                    monitor='val_loss',
                                    verbose=1, save_best_only=True,
                                    save_weights_only=False)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                   patience=self.patience, verbose=1)

        history = CSVLogger(
            os.path.join(self.history_path, self.model_name + "-training_history.csv"),
            append=True
        )
        return save_best, early_stop, history

    # -----------------------------  TRAINING ----------------------------------

    def train_stratified_cv(self):
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        print(self.generator.load_data())



    def train(self, perform_training: bool, stratified_cv: bool):
        self.preprocess()
        self.build_model()

        if perform_training:
            print("-- TRAINING --")
            if stratified_cv:
                self.train_stratified_cv()  # Train model with stratified cross validation

            else:
                save_best, early_stop, history = self.save_model()

                self.model.fit(self.generator.training,
                          epochs=self.epochs,
                          steps_per_epoch=self.generator.training.steps_per_epoch,
                          validation_data=self.generator.validation,
                          validation_steps=self.generator.validation.steps_per_epoch,
                          callbacks=[save_best, history, early_stop])

        # use pre-trained model by changing the model_name to wanted model
        else:
            self.model_name = 'Arch-vgg16_2023-06-19_11:48:01_Stations_config_nr-1_Epochs-50_Loss-categoricalCrossentropy_ImageSize-256_BatchSize-32_Augmentation-True_ValPercent-20'
