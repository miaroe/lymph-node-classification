"""
Contains custom [mdi](https://github.com/androst/medical-data-importers)
importers for the project
"""

import cv2

from mdi.importers import ImageImporter, MetaImageImporter


class PNGImporter(ImageImporter):

    def __init__(self, filepath, *args, **kwargs):
        super(PNGImporter, self).__init__(filepath, *args, **kwargs)

    def load_image(self, filepath):
        img = cv2.imread(filepath)#, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img


def load_png_file(filepath):
    importer = PNGImporter(filepath)
    return importer.load_image(filepath)
