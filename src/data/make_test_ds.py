import os
import random
import shutil

from src.data.db.db_make_new_datastructure import get_dirname_label_map, copy_directory
from src.resources.config import *

def count_files_in_subdirectories(directory):
    total_files = 0

    for root, dirs, files in os.walk(directory):
        total_files += len(files)

    return total_files

# Use after running db_make_new_datastructure.py, will work for both baseline and combined_baseline
def baseline_test_ds():
    # remove old data structure
    if os.path.exists(test_ds_path):
        shutil.rmtree(test_ds_path)

    num_images = count_files_in_subdirectories(data_path)
    test_size = int(test_split * num_images)
    station_test_size = int(test_size / num_stations)
    print('test size: ', test_size)
    print('station test size: ', station_test_size)
    for station in os.listdir(data_path):
        station_path = os.path.join(data_path, station)
        if os.path.isdir(station_path):
            station_frames = random.sample(os.listdir(station_path), station_test_size)
            if not os.path.exists(os.path.join(test_ds_path, station)):
                os.makedirs(os.path.join(test_ds_path, station))
            # move station_frames to test_ds_path with same folder structure
            for frame in station_frames:
                frame_path = os.path.join(station_path, frame)
                test_frame_path = os.path.join(test_ds_path, station, frame)
                os.rename(frame_path, test_frame_path)
                print('moved: ', frame_path, ' to ', test_frame_path)

#baseline_test_ds() # generate test dataset from baseline data structure

def baseline_test_ds_from_patient_list():
    baseline_test_patients = ['Patient_005', 'Patient_016', 'Patient_024', 'Patient_036']

    # remove old data structure
    if os.path.exists(test_ds_path):
        shutil.rmtree(test_ds_path)

    frame_number_dict = {'4L': 0,
                         '4R': 0,
                         '7L': 0,
                         '7R': 0,
                         '10L': 0,
                         '10R': 0,
                         '11L': 0,
                         '11R': 0
                         }
    sequence_label_map = get_dirname_label_map(model_type, station_config_nr)

    for dirname, label in sequence_label_map.items():
        #check if the dirname contains any of the test patients strings and if so, copy the directory to the test_ds_path
        if any(test_patient in dirname for test_patient in baseline_test_patients):
            print(dirname, " is a test patient")
            new_dir = os.path.join(test_ds_path, label)  # (data_path + '_new', label)

            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                frame_number_dict[label] = 0

            print(dirname, " to ", new_dir)
            frame_number_dict = copy_directory(dirname, new_dir, label, frame_number_dict)
            print("Copied: ", dirname, " to ", new_dir)
            print(frame_number_dict)

        else:
            continue

baseline_test_ds_from_patient_list() # generate test dataset from EBUS_Levanger dictionary



def sequence_test_ds():
    # remove old data structure
    if os.path.exists(test_ds_path):
        shutil.rmtree(test_ds_path)
    # move test_split of the patient folders into test_ds_path
    num_patients = len(os.listdir(data_path))
    print('num patients: ', num_patients)
    test_size = int(test_split * num_patients)
    print('test size: ', test_size)
    patient_dirs = os.listdir(data_path)
    patient_dirs = random.sample(patient_dirs, test_size)
    if not os.path.exists(test_ds_path):
        os.makedirs(test_ds_path)
    for patient_dir in patient_dirs:
        patient_path = os.path.join(data_path, patient_dir)
        test_patient_path = os.path.join(test_ds_path, patient_dir)
        os.rename(patient_path, test_patient_path)
        print('moved: ', patient_path, ' to ', test_patient_path)

#sequence_test_ds() # generate test dataset

