import tensorflow_addons as tfa
import tensorflow as tf

loss_dict = {"categoricalCrossentropy": tf.keras.losses.CategoricalCrossentropy(),
                "focalLoss": tfa.losses.SigmoidFocalCrossEntropy()}

def get_loss(loss_name):
    return loss_dict[loss_name]
