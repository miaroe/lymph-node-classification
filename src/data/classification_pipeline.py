import os
import logging

import numpy as np
from mlmia import DatasetPipeline, TaskType
from src.resources.config import set_stations_config

from src.utils.importers import load_png_file


log = logging.getLogger()

class EBUSClassificationPipeline(DatasetPipeline):

    def __init__(self, image_shape=(256, 256), station_config_nr=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.folder_pattern = ["subjects", "stations", "frames"]
        self.filename_pattern = "frame_*.png"
        self.split_by = "subjects"
        self.output_depth = "frames"

        self.task_type = TaskType.CLASSIFICATION

        self.data_loader.inputs_pattern = ["height", "width", "channel"]
        self.data_loader.targets_pattern = ["stations"]
        self.data_loader.apply_resize(height=image_shape[0], width=image_shape[1], apply_to=(0,))

        self.stations_config = set_stations_config(station_config_nr=station_config_nr)
        self.set_stations_to_label()
        self.num_stations = self.get_num_stations()


    def loader_function(self, filepath, index=None, *args, **kwargs):

        # Want image and LN station
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
        station_name = os.path.split(dirname)[1] #Station_10L_001
        station_label = station_name.split('_')[1]  # split ['Station', '10R', '001']

        targets = np.zeros(shape=self.num_stations, dtype=np.uint8) # [0 0 0 0 0 0 0] for LABEL_CONFIG=3
        targets[self.get_station(station_label)] = 1 #[0 0 0 0 0 1 0] for station_name = 10L

        return inputs, targets

    def get_station(self, label):
        try:
            return self.stations_config[label]  # returns if label number is in configuration
        except KeyError:
            return self.stations_config['other']  # else, return class number for 'other'

    def set_stations_to_label(self):
        stations_to_label = {}
        for k, v in self.stations_config.items():
            stations_to_label[v] = k
        self.stations_to_label_dict = stations_to_label

    def get_num_stations(self):
        if hasattr(self, 'stations_config'):
            return len(self.stations_config.keys())
        else:
            raise ValueError("Class has no stations config, and cannot determine number of stations")

