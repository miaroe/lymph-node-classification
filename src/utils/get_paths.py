import os
import pandas as pd

def get_baseline_station_paths(data_path):
    train_paths = []
    val_paths = []
    test_paths = []
    for folder in os.listdir(data_path):
        path = os.path.join(data_path, folder)
        if os.path.isdir(path):
            for station in os.listdir(path):
                station_path = os.path.join(path, station)
                for frame in os.listdir(station_path):
                    frame_path = os.path.join(station_path, frame)
                    if folder == 'train':
                        train_paths.append(frame_path)
                    elif folder == 'val':
                        val_paths.append(frame_path)
                    elif folder == 'test':
                        test_paths.append(frame_path)
    return train_paths, val_paths, test_paths

def get_quality_dataframes(data_path, test=True):
    train_quality_df = pd.read_csv(os.path.join(data_path, 'train_dirname_good_quality_df.csv'))
    val_quality_df = pd.read_csv(os.path.join(data_path, 'val_dirname_good_quality_df.csv'))
    test_quality_df = pd.read_csv(os.path.join(data_path, 'test_dirname_good_quality_df.csv'))
    if test:
        return train_quality_df, val_quality_df, test_quality_df
    else:
        return train_quality_df, val_quality_df


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

def get_frame_paths(station_path, model_type):
    frame_paths_list = []
    frame_names = os.listdir(station_path)
    if model_type == 'sequence': sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].replace('.png', '')))
    else: sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].replace('.hdf5', '')))
    for frame in sorted_frame_names:
        frame_path = os.path.join(station_path, frame)
        if os.path.isfile(frame_path):
            frame_paths_list.append(frame_path)
    return frame_paths_list

def count_number_of_training_samples(data_path):
    train_paths, val_paths = get_training_station_paths(data_path)
    return len(train_paths), len(val_paths)