import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
import pickle
from src.resources.config import *
from src.resources.loss import get_loss
from src.resources.ml_models import get_arch
from src.utils.importers import load_png_file


# TODO: is this enough or do I have to use the same preprocessing as in the training?

def crop_all_images():
    for image_name in os.listdir(local_full_video_path):
        if image_name.endswith(".png"):
            img = tf.io.read_file(os.path.join(local_full_video_path, image_name))
            img = tf.image.decode_png(img, channels=3)
            img = img[100:1035, 530:1658]

            #resize
            img = tf.image.resize(img, [256, 256])

            tf.io.write_file(os.path.join(local_full_video_path, image_name), tf.image.encode_png(img))
            print(image_name, "cropped")

#crop_all_images()

def preprocess_image(image):
    image = tf.image.resize(image, [256, 256])  # resize image
    image = (image / 127.5) - 1.0  # normalize image
    return image


def make_frame_pred_dict():
    model = tf.keras.models.load_model(os.path.join(local_model_path, local_model_name))
    frame_pred_dict = {}

    for image_name in os.listdir(local_full_video_path):
        if image_name.endswith(".png"):
            img = tf.io.read_file(os.path.join(local_full_video_path, image_name))
            img = tf.image.decode_png(img, channels=3)
            img = preprocess_image(img)
            img = tf.expand_dims(img, 0)

            prediction = model.predict(img)
            frame_pred_dict[image_name] = prediction
            print(image_name, prediction)

    # save dict to file with pickle
    with open(os.path.join(local_full_video_path, "frame_pred_dict.pickle"), "wb") as f:
        pickle.dump(frame_pred_dict, f)


make_frame_pred_dict()