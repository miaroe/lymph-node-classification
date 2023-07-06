import os
import numpy as np
from src.resources.config import *

def count_station_distribution(dataset, num_stations):
    count_array = np.zeros(shape=num_stations, dtype=int)

    for _, labels in dataset:
        for label in labels:
            count_array[label.numpy().argmax()] += 1

    return count_array
