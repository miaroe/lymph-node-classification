import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mlmia import TaskType
from mlmia.generator import GeneratorContainer
from src.data.classification_pipeline import EBUSClassificationPipeline

data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger'
save_model_path = '/home/miaroe/workspace/lymph-node-classification/output/models'


# ================================
# Define training parameters
# ================================

BATCH_SIZE = 8
IMG_SIZE = 256
LABEL_CONFIG = 1  # class configuration (gives mapping and mapped labels)
validation_split = 0.2

# Set up pipeline
pipeline = EBUSClassificationPipeline(data_path=data_path,
                                           image_shape=(IMG_SIZE, IMG_SIZE),
                                           tf_dataset=True,
                                           validation_split=validation_split,
                                           batch_size=BATCH_SIZE,
                                           split_by='recordings',
                                           samples_per_load=10,
                                           label_config_nbr=LABEL_CONFIG,
                                           )

for idx, generator in enumerate(pipeline.generators):
    print("Training:", generator.training.get_subjects())
    print("Validation:", generator.validation.get_subjects())
    #print("Testing:", generator.testing.get_subjects())

    inputs, targets = generator.training.next_batch()
    print("Inputs\n--------")

    print("Shape: {}, Pattern: {}, DType: {}, Min: {}, Max: {}\n".
          format(inputs.shape, pipeline.data_loader.inputs_pattern, inputs.dtype, np.min(inputs), np.max(inputs)))
    print("Targets\n--------")
    print("Shape: {}, Pattern: {}, DType: {}, Min: {}, Max: {}\n".
          format(targets.shape, pipeline.data_loader.targets_pattern, inputs.dtype, np.min(targets), np.max(targets)))

pipeline.preview_training_batch(task_type=TaskType.CLASSIFICATION,
                                class_to_label={1: '4L', 2: '4R',
                                                3: '7L', 4: '7R',
                                                5: '10L', 6: '10R',
                                                0: 'other', 7: 'other'})

