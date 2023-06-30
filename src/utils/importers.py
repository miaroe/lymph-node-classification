"""
Contains custom [mdi](https://github.com/androst/medical-data-importers)
importers for the project
"""

import cv2
import os
import numpy as np
import time
from skimage.restoration import denoise_bilateral
from skimage.filters import gaussian
from mdi.importers import ImageImporter, MetaImageImporter
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from skimage.util import random_noise


class PNGImporter(ImageImporter):

    def __init__(self, filepath, *args, **kwargs):
        super(PNGImporter, self).__init__(filepath, *args, **kwargs)

    def load_image(self, filepath):
        img = cv2.imread(filepath, 0) #returns empty matrix if file cannot be read, shape is (1080, 1920)
        return img


def load_png_file(filepath):
    importer = PNGImporter(filepath)
    return importer.load_image(filepath)

def bilateral_filter(img):
    # Measure the execution time
    start_time = time.time()

    denoised = denoise_bilateral(img, win_size=5, sigma_color=0.1, sigma_spatial=1, channel_axis=-1)
    #denoised = cv2.bilateralFilter(img, d=10, sigmaColor=100, sigmaSpace=100)

    # Calculate the execution time
    execution_time = time.time() - start_time
    #print(f"Denoising completed in {execution_time:.3f} seconds")

    return denoised


def gaussian_filter(img):
    gaussian_blur = gaussian(img, sigma=0.5, channel_axis=-1)
    #gaussian_blur = cv2.GaussianBlur(img,(5,5),sigmaX=0)
    return gaussian_blur



# only for testing purposes
# import a random frame into utils
def main():
    file = "frame_30.png"
    assert os.path.exists(file)
    img = cv2.imread(file, 0)
    # Crop image
    img = img[100:1035, 530:1658]
    inputs = (img[..., None] / 255.0).astype(np.float32)
    inputs = inputs * np.ones(shape=(*img.shape, 3))  # 3 ch as input to classification network

    #add filter
    inputs_denoised = bilateral_filter(inputs)
    inputs_gauss = gaussian_filter(inputs)

    # Reshape 'inputs_denoised' to match the shape of 'img'
    #inputs_denoised_reshaped = np.expand_dims(inputs_denoised, axis=-1)

    # Compute PSNR as an indication of image quality
    sigma = 0.12
    noisy = random_noise(inputs, var=sigma ** 2)

    print(f"PSNR noisy: {peak_signal_noise_ratio(inputs, noisy):.3f}")
    print(f"PSNR bilateral: {peak_signal_noise_ratio(inputs, inputs_denoised):.3f}")
    print(f"PSNR gaussian: {peak_signal_noise_ratio(inputs, inputs_gauss):.3f}")

    print(f"mse_noise: {mean_squared_error(inputs, noisy):.3f}")
    print(f"mse_bilateral: {mean_squared_error(inputs, inputs_denoised):.3f}")
    print(f"mse_gaussian: {mean_squared_error(inputs, inputs_gauss):.3f}")

    #compare noise
    import matplotlib.pyplot as plt
    plt.imshow(inputs_denoised, cmap='gray')
    plt.title("Denoised image")
    plt.show()
    plt.imshow(inputs, cmap='gray')
    plt.title("Original image")
    plt.show()
    plt.imshow(inputs_gauss, cmap='gray')
    plt.title("Gaussian filtered image")
    plt.show()





