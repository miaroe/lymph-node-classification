import os

def get_training_station_paths(data_path):
    train_paths = []
    val_paths = []
    for folder in os.listdir(data_path):
        path = os.path.join(data_path, folder)
        if os.path.isdir(path):
            for patient in os.listdir(path):
                patient_path = os.path.join(path, patient)
                for station in os.listdir(patient_path):
                    station_path = os.path.join(patient_path, station)
                    if folder == 'train':
                        train_paths.append(station_path)
                    elif folder == 'val':
                        val_paths.append(station_path)
    return train_paths, val_paths

def get_test_station_paths(data_path):
    test_paths = []
    for patient in os.listdir(data_path):
        patient_path = os.path.join(data_path, patient)
        if os.path.isdir(patient_path):
            for station in os.listdir(patient_path):
                station_path = os.path.join(patient_path, station)
                test_paths.append(station_path)
    return test_paths

def get_frame_paths(station_path):
    frame_paths_list = []
    frame_names = os.listdir(station_path)
    sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].replace('.png', '')))
    for frame in sorted_frame_names:
        frame_path = os.path.join(station_path, frame)
        if os.path.isfile(frame_path):
            frame_paths_list.append(frame_path)
    return frame_paths_list

def count_number_of_training_samples(data_path):
    train_paths, val_paths = get_training_station_paths(data_path)
    return len(train_paths), len(val_paths)