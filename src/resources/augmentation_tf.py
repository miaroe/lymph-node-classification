import tensorflow as tf
from tensorflow.keras import layers


class RandomAugmentation(layers.Layer):
    def __init__(self, augmentation_layers, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.augmentation_layers = augmentation_layers
        self.probability = probability

    def call(self, data, **kwargs):
        for augmentation_layer in self.augmentation_layers:
            apply_augmentation = tf.random.uniform(shape=(), dtype=tf.float32) < self.probability
            #print('apply_augmentation: ', apply_augmentation)
            if apply_augmentation:
                augmentation_layer.randomize()
                data = augmentation_layer(data)
        return data


class GammaTransform(layers.Layer):
    def __init__(self, low=0.8, high=1.2, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high
        self._random_gamma = None

    def randomize(self):
        self._random_gamma = tf.random.uniform(shape=(), minval=self.low, maxval=self.high)

    def call(self, data, **kwargs):
        #self.randomize()
        #print('GammaTransform: ', self._random_gamma)
        return tf.clip_by_value(tf.pow(data, self._random_gamma), -1, 1)


class ContrastScale(layers.Layer):
    def __init__(self, min_scale=0.25, max_scale=1.75, **kwargs):
        super().__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._random_scale = None

    def randomize(self):
        self._random_scale = tf.random.uniform(shape=(), minval=self.min_scale, maxval=self.max_scale)

    def call(self, data, **kwargs):
        #self.randomize()
        #print('ContrastScale: ', self._random_scale)
        return tf.clip_by_value(data * self._random_scale, -1, 1)


class Blur(layers.Layer):
    def __init__(self, sigma_min=0.0, sigma_max=0.8, **kwargs):
        super().__init__(**kwargs)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self._random_sigma = None

    def randomize(self):
        self._random_sigma = tf.random.uniform(shape=(), minval=self.sigma_min, maxval=self.sigma_max)

    def call(self, data, *args, **kwargs):
        #self.randomize()
        #print('Blur: ', self._random_sigma)
        return tf.clip_by_value(tf.image.gaussian_filter2d(data, [3, 3], self._random_sigma), -1, 1) # not sure about this


class BrightnessTransform(layers.Layer):
    def __init__(self, max_scale=0.2, **kwargs):
        super().__init__(**kwargs)
        self.max_scale = max_scale
        self._random_scale = None

    def randomize(self):
        self._random_scale = tf.random.uniform(shape=(), minval=-self.max_scale, maxval=self.max_scale)

    def call(self, data, **kwargs):
        #self.randomize()
        #print('BrightnessTransform: ', self._random_scale)
        return tf.clip_by_value(data + self._random_scale, -1, 1)


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

        self._random_sigma_x = None
        self._random_sigma_y = None
        self._random_strength = None
        self._random_location = None

    def randomize(self):
        self._random_sigma_x = np.random.uniform(self.sigma_x[0], self.sigma_x[1], 1).astype(np.float32)
        self._random_sigma_y = np.random.uniform(self.sigma_y[0], self.sigma_y[1], 1).astype(np.float32)
        self._random_strength = np.random.uniform(self.strength[0], self.strength[1], 1).astype(np.float32)
        self._random_location = np.random.uniform(-1.0, 1.0, 2).astype(np.float32)

    def call(self, data, return_map=False, **kwargs):
        #self.randomize()
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
