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


def get_training_station_paths(data_path, model_type):
    train_paths = []
    val_paths = []
    if model_type == 'sequence_cv':
        # get paths from train and val csv files
        train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_path, 'val.csv'))
        for index, row in train_df.iterrows():
            train_paths.append(row['dirname'])
        for index, row in val_df.iterrows():
            val_paths.append(row['dirname'])
    else:
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

def get_test_station_paths(data_path, model_type):
    test_paths = []
    if model_type == 'sequence_cv':
        # get paths from train and val csv files
        train_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        for index, row in train_df.iterrows():
            test_paths.append(row['dirname'])
    else:
        for patient in os.listdir(os.path.join(data_path, 'test')):
            patient_path = os.path.join(data_path, patient)
            if os.path.isdir(patient_path):
                for station in os.listdir(patient_path):
                    station_path = os.path.join(patient_path, station)
                    test_paths.append(station_path)
    return test_paths

def get_frame_paths(station_path, model_type):
    frame_paths_list = []
    frame_names = os.listdir(station_path)
    if model_type == 'sequence' or model_type == 'sequence_cv': sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].replace('.png', '')))
    else: sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].replace('.hdf5', '')))
    for frame in sorted_frame_names:
        frame_path = os.path.join(station_path, frame)
        if os.path.isfile(frame_path):
            frame_paths_list.append(frame_path)
    return frame_paths_list

def count_number_of_training_samples(data_path, fold, model_type):
    if fold is not None:
        data_path = os.path.join(data_path, f'fold_{fold}_v2')
    train_paths, val_paths = get_training_station_paths(data_path, model_type)
    return len(train_paths), len(val_paths)