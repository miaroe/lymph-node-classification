# https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html

import scipy.stats
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from src.resources.config import get_stations_config
from src.resources.loss import get_loss
from src.resources.ml_models import get_arch

model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/2023-11-25/17:12:56/'
num_test_images = 2210
num_stations = 8
model_arch = 'cvc_net'

stations_config = get_stations_config(3)
print(list(stations_config.keys()))

test_ds = tf.keras.utils.image_dataset_from_directory(
            directory='/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/baseline/Levanger_and_StOlavs/test',
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            class_names=list(stations_config.keys()),
            batch_size=32,
            image_size=(256, 256),
            shuffle=False
        )

rescale = tf.keras.Sequential([
            #tf.keras.layers.Rescaling(1. / 127.5, offset=-1)  # specific for mobilenet TODO: change for other models
            tf.keras.layers.Rescaling(1. / 255.0)
        ])

test_ds = test_ds.map(lambda x, y: (rescale(x), y))

model = get_arch(model_arch, instance_size=(256, 256, 3), num_stations=num_stations, stateful=False)

model.compile(loss=get_loss('categoricalCrossEntropy'), optimizer='adam',
              metrics=['accuracy', Precision(), Recall()])

model.load_weights(filepath=os.path.join(model_path, 'best_model')).expect_partial()

score = model.evaluate(test_ds, return_dict=True, verbose=1)
acc_test = score['accuracy']
print('acc_test: ', acc_test)

confidence = 0.95  # Change to your desired confidence level
z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)

ci_length = z_value * np.sqrt((acc_test * (1 - acc_test)) / num_test_images)

ci_lower = acc_test - ci_length
ci_upper = acc_test + ci_length

print('ci_length: ', ci_length)
print('ci_lower: ', ci_lower)
print('ci_upper: ', ci_upper)




