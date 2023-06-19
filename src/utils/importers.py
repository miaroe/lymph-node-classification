"""
Contains custom [mdi](https://github.com/androst/medical-data-importers)
importers for the project
"""

import cv2
import os
import numpy as np

from mdi.importers import ImageImporter, MetaImageImporter


class PNGImporter(ImageImporter):

    def __init__(self, filepath, *args, **kwargs):
        super(PNGImporter, self).__init__(filepath, *args, **kwargs)

    def load_image(self, filepath):
        img = cv2.imread(filepath, 0) #returns empty matrix if file cannot be read, shape is (1080, 1920)
        return img


def load_png_file(filepath):
    importer = PNGImporter(filepath)
    return importer.load_image(filepath)


# only for testing purposes
# import a random frame into utils
def main():
    file = "frame_56.png"
    assert os.path.exists(file)
    img = cv2.imread(file, 0)
    # Crop image
    img = img[100:1035, 530:1658]
    inputs = (img / 255.0).astype(np.float32)
    print(inputs.shape) #grayscale
    for i in range(650,700):
        print(inputs[i][i])

#main()
