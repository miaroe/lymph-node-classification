import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.losses import Loss
import numpy as np

class MaskedMeanAbsoluteError(Loss):

    def __init__(self, name="masked_mae"):
        super(MaskedMeanAbsoluteError, self).__init__(name=name)

    def call(self, y_true, y_pred, mask_value=-1):
        mask = y_true != mask_value
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)


class MaskedBinaryCrossEntropy(Loss):
    def __init__(self, name="masked_binary_crossentropy"):
        super(MaskedBinaryCrossEntropy, self).__init__(name=name)

    def call(self, y_true, y_pred, mask_value=-1):
        mask = y_true != mask_value
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        return tf.keras.metrics.binary_crossentropy(y_true, y_pred)


class MaskedCategoricalCrossEntropy(Loss):
    def __init__(self, name="masked_categorical_crossentropy"):
        super(MaskedCategoricalCrossEntropy, self).__init__(name=name)

    def call(self, y_true, y_pred, mask_value=-1):
        mask = y_true != mask_value
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        loss = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
        return loss


def get_loss(loss_name):
    loss_dict = {
        "categoricalCrossEntropy": tf.keras.losses.CategoricalCrossentropy(),
        "focalLoss": tfa.losses.SigmoidFocalCrossEntropy(),
        "maskedMeanAbsoluteError": MaskedMeanAbsoluteError(),
        "maskedBinaryCrossEntropy": MaskedBinaryCrossEntropy(),
        "maskedCategoricalCrossEntropy": MaskedCategoricalCrossEntropy()
    }

    return loss_dict[loss_name]
