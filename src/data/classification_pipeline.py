import os

import numpy as np
from mlmia import DatasetPipeline, TaskType

from src.utils.importers import load_png_file


class EBUSClassificationPipeline(DatasetPipeline):

    def __init__(self, image_shape=(256, 256), views=None, class_config_nbr=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.folder_pattern = ["subjects", "recordings", "files"]
        self.filename_pattern = "frame_*.png"
        self.split_by = "subjects"
        self.output_depth = "files"
        #self.views = ["4R", "4L"] if views is None else views

        self.task_type = TaskType.CLASSIFICATION

        self.data_loader.inputs_pattern = ["height", "width", "channel"]
        self.data_loader.targets_pattern = ["channel"]

        # self.data_loader.apply_cropping(width_range=(530, 1658), height_range=(100, 1035), apply_to=(0,))
        self.data_loader.apply_resize(height=image_shape[0], width=image_shape[1], apply_to=(0,))

        self.set_class_configuration(config_nbr=class_config_nbr)
        self.num_classes = self.get_num_classes()

    def loader_function(self, filepath, index=None, *args, **kwargs):
        # Want image (is .png) and LN station (# 0-9)
        dirname = os.path.dirname(filepath)
        tag = os.path.basename(os.path.splitext(filepath)[0])

        # IMAGE
        image_file = os.path.join(dirname, "{0}.png".format(tag))
        img = load_png_file(image_file)
        # Crop image
        img = img[100:1035, 530:1658]
        # Normalize image
        inputs = (img[..., None]/255.0).astype(np.float32)
        inputs = inputs * np.ones(shape=(*img.shape, 3))  # 3 ch as input to classification network

        # LN STATION
        sequence_name = os.path.split(dirname)[1]
        station_label = sequence_name.split('_')[1]  # split ['Station', 'XXL/R', '001']

        targets = np.zeros(shape=self.num_classes, dtype=np.uint8)
        targets[self.get_class(station_label)] = 1

        return inputs, targets

    def get_num_classes(self):
        if hasattr(self, 'class_config'):
            return len(self.class_config.keys())
        else:
            raise ValueError("Class has no class config, and cannot determine number of classes")

    # ======================================
    # Define mappings from old to new labels
    # ======================================
    # Label lymph nodes in ascending order, and L before R stations
    #   e.g. 4L, 4R, 7L, 7R, 10L, 10R, 11L, 11R, ..., then 'lymph_node'
    def get_class(self, label):
        try:
            return self.class_config[label]    # returns if label number is in configuration
        except KeyError:
            return self.class_config['other']  # else, return class number for 'other'

    def set_class_configuration(self, config_nbr):

        if config_nbr == 1:
            self.class_config = {
                'other': 0,
                '4L': 1,
                '4R': 2,
                #'other': 3,
            }
        elif config_nbr == 2:
            self.class_config = {
                'other': 0,
                '4L': 1,
                '4R': 2,
                '7L': 3,
                '7R': 4,
            }
        elif config_nbr == 3:
            self.class_config = {
                'other': 0,
                '4L': 1,
                '4R': 2,
                '7L': 3,
                '7R': 4,
                '10L': 5,
                '10R': 6,
            }
        elif config_nbr == 4:
            self.class_config = {
                'other': 0,
                '4L': 1,
                '4R': 2,
                '7L': 3,
                '7R': 4,
                '10L': 5,
                '10R': 6,
                '11L': 7,
                '11R': 8,
                '7': 9,
            }
        else:
            print("Choose one of the predefined sets of classes: config_nbr={1, 2, 3, 4}")
            exit(-1)

        self.set_class_to_label()

    def set_class_to_label(self):
        class_to_label = {}
        for k, v in self.class_config.items():
            class_to_label[v] = k
        self.class_to_label_dict = class_to_label

    def class_to_label(self, x):
        if x not in self.class_to_label_dict.keys():
            raise IndexError(f'Class {x} not an available class ({self.class_to_label_dict.keys()})')
        return self.class_to_label_dict[x]
