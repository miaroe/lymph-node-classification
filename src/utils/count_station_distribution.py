import os
import numpy as np

def count_station_distribution(pipeline, batch_generator):
    count_array = np.zeros(shape=pipeline.get_num_stations(), dtype=int)

    for filepath in batch_generator.files:
        dirname = os.path.dirname(filepath)
        sequence_name = os.path.split(dirname)[1]
        station_label = sequence_name.split('_')[1]  # split ['Station', '10R', '001']
        count_array[pipeline.get_station(station_label)] += 1
    return count_array