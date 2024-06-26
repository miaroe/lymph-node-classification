import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K

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

class CategoricalFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='categorical_focal_crossentropy'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = - y_true * tf.math.log(y_pred)
        loss = self.alpha * tf.pow(1 - y_pred, self.gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

class CustomLossWithQuality(tf.keras.losses.Loss):
    def __init__(self, name='custom_loss_with_quality'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Assume y_true is a tuple packed with (true_labels, quality)
        # Unpack the true labels and quality scores
        true_labels, quality = y_true[..., :-1], y_true[..., -1]  # Assuming quality is the last column

        # Calculate base loss with the true labels
        base_loss = tf.keras.losses.categorical_crossentropy(true_labels, y_pred)

        # Adjust loss based on quality
        # Example: good quality (1.0) => weight 3.0, bad quality (0.0) => weight 1.0
        quality_weights = 1.0 + (quality * 2.0)
        weighted_loss = base_loss * quality_weights

        return tf.reduce_mean(weighted_loss)


def get_loss(loss_name):
    loss_dict = {
        "categoricalCrossEntropy": tf.keras.losses.CategoricalCrossentropy(),
        "binaryCrossEntropy": tf.keras.losses.BinaryCrossentropy(),
        "focalCrossEntropy": CategoricalFocalCrossEntropy(), # tf.keras.losses.CategoricalFocalCrossentropy( not available for this version of tensorflow
        "maskedMeanAbsoluteError": MaskedMeanAbsoluteError(),
        "maskedBinaryCrossEntropy": MaskedBinaryCrossEntropy(),
        "maskedCategoricalCrossEntropy": MaskedCategoricalCrossEntropy(),
        "customLossWithQuality": CustomLossWithQuality()
    }

    return loss_dict[loss_name]
