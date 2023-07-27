from src.utils.count_station_distribution import count_station_distribution
import numpy as np

# calculate class_weight only for multiclass classification
def get_class_weight(train_ds, num_stations):
    if num_stations > 2:
        # balance data by calculating class weights and using them in fit
        count_array = count_station_distribution(train_ds, num_stations)
        class_weight = {idx: (1 / elem) * np.sum(count_array) / num_stations for idx, elem in enumerate(count_array)}
        print(class_weight)

    else: class_weight = None

    return class_weight