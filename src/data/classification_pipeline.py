import os
import logging

import numpy as np
from mlmia import DatasetPipeline, TaskType

from src.utils.importers import load_png_file


log = logging.getLogger()


class EBUSClassificationPipeline(DatasetPipeline):

    def __init__(self, image_shape=(256, 256), station_config_nr=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.folder_pattern = ["subjects", "stations", "frames"]
        self.filename_pattern = "frame_*.png"
        self.split_by = "subjects"
        self.output_depth = "frames"

        self.task_type = TaskType.CLASSIFICATION

        self.data_loader.inputs_pattern = ["height", "width", "channel"]
        self.data_loader.targets_pattern = ["stations"]
        self.data_loader.apply_resize(height=image_shape[0], width=image_shape[1], apply_to=(0,))

        self.set_stations_config(config_nr=station_config_nr)
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

    # -----------------------------  STATIONS CONFIG ----------------------------------

    # Label lymph nodes in ascending order, and L before R stations
    #   e.g. 4L, 4R, 7L, 7R, 10L, 10R, 11L, 11R, ...

    def set_stations_config(self, config_nr):
        if config_nr == 1:
            self.stations_config = {
                'other': 0,
                '4L': 1,
                '4R': 2,
                # 'other': 3,
            }
        elif config_nr == 2:
            self.stations_config = {
                'other': 0,
                '4L': 1,
                '4R': 2,
                '7L': 3,
                '7R': 4,
            }
        elif config_nr == 3:
            self.stations_config = {
                'other': 0,
                '4L': 1,
                '4R': 2,
                '7L': 3,
                '7R': 4,
                '10L': 5,
                '10R': 6,
            }
        elif config_nr == 4:
            self.stations_config = {
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
            print("Choose one of the predefined sets of stations: config_nbr={1, 2, 3, 4}")
            exit(-1)

        self.set_stations_to_label()

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

    def stations_to_label(self, x):
        if x not in self.stations_to_label_dict.keys():
            raise IndexError(f'Station {x} not an available station ({self.stations_to_label_dict.keys()})')
        return self.stations_to_label_dict[x]