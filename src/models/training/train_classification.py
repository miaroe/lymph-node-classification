import os
import logging

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import Precision, Recall

from src.resources.ml_models import get_arch
from src.data.classification_pipeline import EBUSClassificationPipeline

logger = logging.getLogger()
def train_model(perform_training, data_path, image_shape, tf_dataset, validation_split,
                batch_size, split_by, samples_per_load, label_config, augment_data, model_arch,
                instance_size, learning_rate, save_model_path, history_path, model_name, patience,
                epochs):

    print("Trainer: " + model_arch)
    trainer = BaselineTrainer(data_path, image_shape, tf_dataset, validation_split, batch_size,
                                split_by, samples_per_load, label_config, augment_data, model_arch,
                                instance_size, learning_rate, save_model_path, history_path,
                                model_name, patience, epochs)

    # Perform training
    trainer.train(perform_training)
    return trainer

class BaselineTrainer:

    def __init__(self,
                 data_path: str,
                 image_shape: tuple,
                 tf_dataset: bool,
                 validation_split: float,
                 batch_size: int,
                 split_by: str,
                 samples_per_load: int,
                 label_config: int,
                 augment_data: bool,
                 model_arch: str,
                 instance_size: tuple,
                 learning_rate: float,
                 save_model_path: str,
                 history_path: str,
                 model_name: str,
                 patience: int,
                 epochs: int
                 ):

        self.data_path = data_path
        self.image_shape = image_shape
        self.tf_dataset = tf_dataset
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.split_by = split_by
        self.samples_per_load = samples_per_load
        self.label_config = label_config
        self.augment_data = augment_data
        self.model_arch = model_arch
        self.instance_size = instance_size
        self.learning_rate = learning_rate
        self.save_model_path = save_model_path
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
                                   batch_size=self.batch_size,
                                   split_by=self.split_by,
                                   samples_per_load=self.samples_per_load,
                                   label_config_nbr=self.label_config,
                                   )
        if self.augment_data:
            self.pipeline.data_augmentor.add_rotation(max_angle=30)
            self.pipeline.data_augmentor.add_gamma_transformation(0.5, 1.5)
            self.pipeline.data_augmentor.add_gaussian_shadow()
        '''
        self.pipeline.preview_training_batch(task_type=TaskType.CLASSIFICATION,
                                        class_to_label={1: '4L', 2: '4R',
                                                        3: '7L', 4: '7R',
                                                        5: '10L', 6: '10R',
                                                        0: 'other', 7: 'other'})

        '''
        self.generator = self.pipeline.generator_containers[0]
        print('Training subjects', self.generator.training.get_subjects())
        print('Validation subjects', self.generator.validation.get_subjects())

    # -----------------------------  BUILDING AND SAVING MODEL ----------------------------------

    def build_model(self):
        self.model = get_arch(self.model_arch, self.instance_size, self.pipeline.get_num_stations())
        print(self.model.summary())

        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()],
        )
    def save_model(self):
        os.makedirs(self.save_model_path, exist_ok=True)
        os.makedirs(self.history_path, exist_ok=True)

        model_checkpoint_path = os.path.join(self.save_model_path, self.model_name + ".h5")

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

    def train(self, perform_training: bool):
        self.preprocess()
        self.build_model()

        if perform_training:
            print("-- TRAINING --")
            save_best, early_stop, history = self.save_model()

            self.model.fit(self.generator.training,
                      epochs=self.epochs,
                      steps_per_epoch=self.generator.training.steps_per_epoch,
                      validation_data=self.generator.validation,
                      validation_steps=self.generator.validation.steps_per_epoch,
                      callbacks=[save_best, history, early_stop])

        # use pre-trained model by changing the model_name to wanted model
        else:
            self.model_name = 'EBUSClassification_Stations-7_2023-06-07-093431_Epochs-1_ImageSize-256_BatchSize-32_Augmentation-False_ValPercent-20'
