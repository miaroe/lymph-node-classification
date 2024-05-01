import numpy as np
from tensorflow.keras import layers
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
import tensorflow as tf

class RandomAugmentation(layers.Layer):
    def __init__(self, augmentation_layers, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.augmentation_layers = augmentation_layers
        self.probability = probability

    def call(self, data, **kwargs):
        for augmentation_layer in self.augmentation_layers:
            apply_augmentation = np.random.random() < self.probability
            #apply_augmentation = True
            if apply_augmentation:
                #print('apply augmentation: ', augmentation_layer)
                augmentation_layer.randomize()
                data = augmentation_layer(data)
        return data

class RandomAugmentationSequence(layers.Layer):
    def __init__(self, augmentation_layers, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.augmentation_layers = augmentation_layers
        self.probability = probability

    def call(self, data, **kwargs):
        for augmentation_layer in self.augmentation_layers:
            apply_augmentation = np.random.random() < self.probability
            #apply_augmentation = True
            if apply_augmentation:
                #print('apply augmentation: ', augmentation_layer)
                augmentation_layer.randomize()
                # Use tf.map_fn to apply augmentation to each element of the data tensor
                data = tf.map_fn(lambda x: augmentation_layer(x), data, dtype=tf.float32)

        return data


class GammaTransform(layers.Layer):
    def __init__(self, low=0.8, high=1.2, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high
        self._random_gamma = high

    def randomize(self):
        self._random_gamma = np.random.uniform(self.low, self.high)

    def call(self, data, **kwargs):
        #print('GammaTransform: ', self._random_gamma)
        return np.clip(np.power(data, self._random_gamma), 0, 1)


class ContrastScale(layers.Layer):
    def __init__(self, min_scale=0.25, max_scale=1.75, **kwargs):
        super().__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._random_scale = max_scale

    def randomize(self):
        self._random_scale = np.random.uniform(self.min_scale, self.max_scale)

    def call(self, data, **kwargs):
        #print('ContrastScale: ', self._random_scale)
        return np.clip(data * self._random_scale, 0, 1)


class Blur(layers.Layer):
    def __init__(self, sigma_min=0.0, sigma_max=0.8, **kwargs):
        super().__init__(**kwargs)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self._random_sigma = sigma_max

    def randomize(self):
        self._random_sigma = np.random.uniform(self.sigma_min, self.sigma_max)

    def call(self, data, *args, **kwargs):
        #print('Blur: ', self._random_sigma)
        return gaussian_filter(data, self._random_sigma) * np.array(data > 0)


class BrightnessTransform(layers.Layer):
    def __init__(self, max_scale=0.2, **kwargs):
        super().__init__(**kwargs)
        self.max_scale = max_scale
        self._random_scale = max_scale

    def randomize(self):
        self._random_scale = np.random.uniform(-self.max_scale, self.max_scale)

    def call(self, data, **kwargs):
        #print('BrightnessTransform: ', self._random_scale)
        if np.all(data == 0): #check if images are all zeros
            return data
        return np.clip(data + self._random_scale, 0, 1)


class GaussianShadow(layers.Layer):
    """
    Parameters
    ----------
    sigma_x: tuple,
        Sigma value in x-direction with minmax as tuple, (min, max)
    sigma_y: tuple
        Sigma value in y-direction with minmax as tuple, (min, max)
    strength: tuple
        Signal strength with minmax as tuple, (min, max)
    location: tuple, optional
        Force location of shadow at specific location (x, y). Location (x, y) is given as a fraction of the image size,
        i.e. between 0 and 1.
    """

    def __init__(self, sigma_x: tuple, sigma_y: tuple, strength: tuple, location: tuple = None, **kwargs):
        super().__init__(**kwargs)

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.strength = strength
        self.location = location

        self._random_sigma_x = 0.5
        self._random_sigma_y = 0.9
        self._random_strength =  0.8
        self._random_location =  np.random.uniform(-1.0, 1.0, 2).astype(np.float32)

    def randomize(self):
        self._random_sigma_x = np.random.uniform(self.sigma_x[0], self.sigma_x[1], 1).astype(np.float32)
        self._random_sigma_y = np.random.uniform(self.sigma_y[0], self.sigma_y[1], 1).astype(np.float32)
        self._random_strength = np.random.uniform(self.strength[0], self.strength[1], 1).astype(np.float32)
        self._random_location = np.random.uniform(-1.0, 1.0, 2).astype(np.float32)

    def call(self, data, return_map=False, **kwargs):
        #print('GaussianShadow: ', self._random_sigma_x, self._random_sigma_y, self._random_strength, self._random_location)
        x, y = np.meshgrid(np.linspace(-1, 1, data.shape[0], dtype=np.float32),
                           np.linspace(-1, 1, data.shape[1], dtype=np.float32), copy=False, indexing='ij')

        if self.location:
            x_mu, y_mu = self.location[0] * 2 - 1, self.location[1] * 2 - 1
        else:
            x_mu, y_mu = self._random_location

        g = 1.0 - self._random_strength * np.exp(-((x - x_mu) ** 2 / (2.0 * self._random_sigma_x ** 2)
                                                   + (y - y_mu) ** 2 / (2.0 * self._random_sigma_y ** 2)))

        augmented_data = np.copy(data)
        if len(augmented_data.shape) > 2:
            augmented_data = augmented_data * g[..., None]
        else:
            augmented_data = augmented_data * g

        if return_map:
            return augmented_data, g
        return augmented_data


class Rotation(layers.Layer):
    """
    Rotates image with arbitrary angle in bounds [-max_angle:max_angle]

    Parameters
    ----------
    max_angle: number
        Maximum angle in degrees.
    """

    def __init__(self, max_angle=15, flow_indices=None, segmentation_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.max_angle = max_angle
        self._random_angle = max_angle
        self.flow_indices = flow_indices
        self.segmentation_indices = segmentation_indices

    def randomize(self):
        self._random_angle = np.random.randint(-self.max_angle, self.max_angle)

    def call(self, data, data_index=None, *args, **kwargs):
        #print('Rotation: ', self._random_angle)
        data = np.copy(data)

        if len(data.shape) > 2:
            for channel in range(0, data.shape[2]):
                if self.segmentation_indices is not None and data_index in self.segmentation_indices:
                    cval = find_best_padding_cval(data[..., channel])
                    data[..., channel] = rotate(data[..., channel], self._random_angle, order=0, reshape=False,
                                                cval=cval)
                else:
                    data[..., channel] = rotate(data[..., channel], self._random_angle, order=1, reshape=False)
        else:
            data = rotate(data, self._random_angle, order=1, reshape=False)

        if self.flow_indices is not None and data_index in self.flow_indices:
            theta = np.radians(-self._random_angle)
            c, s = np.cos(theta), np.sin(theta)
            rot_mat = np.array(((c, -s), (s, c)))
            data = data.dot(rot_mat)
        return data

class NonLinearMap(layers.Layer):
    def __init__(self, alpha=0.544559, beta=1.686562, gamma=5.598193, delta=0.638681, y0=0.002457387314, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.y0 = y0


    def call(self, data, *args, **kwargs):
        #print('NonLinearMap: ', self.alpha, self.beta, self.gamma, self.delta, self.y0)
        data = self.alpha * np.exp(-np.exp(self.beta - self.gamma * data) + self.delta * data) - self.y0
        return np.clip(data, a_min=0, a_max=1)

    def randomize(self):
        pass

def find_best_padding_cval(segmentation_mask):
    """
    To be used geometrical transformations (scale, rotation, translation...) on segmentation masks.
    Indeed, using filling modes like 'nearest' will lead to irrealistic shapes in case the masked structure
    (background, left_ventricle, left-atrium...) is at the border of the frame. Padding with 'constant' to 0 is not a
    solution either since we want to fill with continuity with the background channel (which is mostly on the borders
    of the image).
    This function finds the best fill value for each channel (value which is majority at the four images edges) of the
    segmentation mask.
    """
    edge1 = segmentation_mask[0, :-1]
    edge2 = segmentation_mask[:-1, -1]
    edge3 = segmentation_mask[-1, 1:]
    edge4 = segmentation_mask[1:, 0]
    average_edge = np.mean(np.concatenate([edge1, edge2, edge3, edge4]))
    return np.round(average_edge).astype(segmentation_mask.dtype)


