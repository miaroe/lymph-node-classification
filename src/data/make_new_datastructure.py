import shutil
import numpy as np
import cv2
import pandas as pd
import random

from src.data.db.db_dirname_label_map import get_dirname_label_map_db
from src.resources.config import *
from src.data.db.db_image_quality import get_dirname_quality_map
from alive_progress import alive_bar

#------------------ Helper functions ------------------#

def get_subject_ids(dirname_label_df, test_patient_ids=None):
    """
    chooses which patients to use for training and validation at random according to subject_id column and validation_split
    if test_patient_ids is not None, then the test_patient_ids are used for testing and the rest for training and validation
    else, the test_patient_ids are chosen at random according to test_split

    :param dirname_label_df:
    :param test_patient_ids:
    :return:
    """

    unique_patient_ids = dirname_label_df['patient_id'].unique()
    np.random.shuffle(unique_patient_ids)
    num_val_patients = int(len(unique_patient_ids) * validation_split)
    print('num_val_patients: ', num_val_patients)

    if test_patient_ids is not None: # set test patient ids manually
        unique_patient_ids = [patient_id for patient_id in unique_patient_ids if patient_id not in test_patient_ids]
        train_patient_ids = unique_patient_ids[num_val_patients:]
        val_patient_ids = unique_patient_ids[:num_val_patients]
    else:
        num_test_patients = int(len(unique_patient_ids) * test_split)

        train_patient_ids = unique_patient_ids[num_val_patients + num_test_patients:]
        val_patient_ids = unique_patient_ids[:num_val_patients]
        test_patient_ids = unique_patient_ids[num_val_patients:num_val_patients + num_test_patients]
        #check that all labels are present in train, val and test
        train_labels = dirname_label_df[dirname_label_df['patient_id'].isin(train_patient_ids)]['label'].unique()
        val_labels = dirname_label_df[dirname_label_df['patient_id'].isin(val_patient_ids)]['label'].unique()
        test_labels = dirname_label_df[dirname_label_df['patient_id'].isin(test_patient_ids)]['label'].unique()
        if len(train_labels) != len(val_labels) and len(train_labels) != len(test_labels) and len(val_labels) != len(test_labels):
            raise Exception("Not all labels are present in train, val, and test")

    train_dirname_label_df = dirname_label_df[dirname_label_df['patient_id'].isin(train_patient_ids)]
    val_dirname_label_df = dirname_label_df[dirname_label_df['patient_id'].isin(val_patient_ids)]
    test_dirname_label_df = dirname_label_df[dirname_label_df['patient_id'].isin(test_patient_ids)]

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # save the train, val and test sequence_label_df to csv
    train_dirname_label_df.to_csv(os.path.join(data_path, 'train_dirname_label_df.csv'))
    val_dirname_label_df.to_csv(os.path.join(data_path, 'val_dirname_label_df.csv'))
    test_dirname_label_df.to_csv(os.path.join(data_path, 'test_dirname_label_df.csv'))

    return train_dirname_label_df, val_dirname_label_df, test_dirname_label_df


def crop_image(filename, folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        image = image[100:1035, 530:1658]
        cv2.imwrite(image_path, image)
        print("Cropped: ", image_path)


def crop_all_images_in_path(path, model_type):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if model_type == 'sequence' or model_type == 'combined_sequence':
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                for filename in os.listdir(subfolder_path):
                    crop_image(filename, subfolder_path)
        else:
            for filename in os.listdir(folder_path):
                crop_image(filename, folder_path)

#------------------ EBUS baseline helper functions ------------------#

def create_new_baseline_from_df(df, baseline_dir, dirname_quality_map):
    """
    creates a new data structure for the baseline model
    :param df:
    :param baseline_dir:
    :param dirname_quality_map:
    :return:
    """
    frame_number_dict = {'4L': 0,
                         '4R': 0,
                         '7': 0,
                         '7L': 0,
                         '7R': 0,
                         '10L': 0,
                         '10R': 0,
                         '11L': 0,
                         '11R': 0
                         }
    # loop through train and val sequence_label_df and copy the images to the new data structure
    for index, row in df.iterrows():
        dirname = row['dirname']
        label = row['label']
        new_dir = os.path.join(baseline_dir, label)

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            frame_number_dict[label] = 0

        if filter_data:
            # only copy over images labeled as 'good' or 'ok' quality
            if dirname_quality_map[dirname] != 'poor':
                # copying all files from the source directory to destination directory
                frame_number_dict = copy_directory_baseline(dirname, new_dir, label, frame_number_dict)
                print("Copied: ", dirname, " to ", new_dir)
                print("frame_number_dict: ", frame_number_dict)

        else:
            # copying all files from the source directory to destination directory
            frame_number_dict = copy_directory_baseline(dirname, new_dir, label, frame_number_dict)
            print("Copied: ", dirname, " to ", new_dir)
            print("frame_number_dict: ", frame_number_dict)

def copy_directory_baseline(src_dir, dst_dir, label, frame_number_dict):
    """
    copy sequence into a new dircetory under the label name
    rename the files to the frame number and update the frame_number_dict

    :param src_dir:
    :param dst_dir:
    :param label:
    :param frame_number_dict:
    :return:
    """
    if os.path.isdir(src_dir):
        # rename file names and copy to new directory
        for filename in os.listdir(src_dir):
            src = os.path.join(src_dir, filename)
            dst = os.path.join(dst_dir, str(frame_number_dict[label]) + '.png')
            shutil.copyfile(src, dst)
            frame_number_dict[label] += 1
        return frame_number_dict
    else:
        raise Exception("Directory does not exist")


#------------------ Create EBUS baseline ------------------#

def create_EBUS_baseline():
    """
    to make new data structure from EBUS_Levanger (and EBUS_StOlavs if combined model)
    - First the subject ids to use for training and validation are retrieved by skipping the test patients (get_subject_ids)
    - Then the new data structure is created from the train, val and test sequence_label_df (create_new_datastructure_from_df)
    - At the end all images in the new data structure are cropped (crop_all_images_in_path)
    model_type, station_config_nr, filter_data, data_path and validation_split are defined in config.py

    :return:
    """
    baseline_train_dir = os.path.join(data_path, 'train')
    baseline_val_dir = os.path.join(data_path, 'val')

    #'Patient_005', 'Patient_016', 'Patient_024', 'Patient_036'
    baseline_test_subject_ids = [715, 724, 735, 743]

    # remove old data structure
    if os.path.exists(baseline_train_dir):
        shutil.rmtree(baseline_train_dir)

    if os.path.exists(baseline_val_dir):
        shutil.rmtree(baseline_val_dir)

    if os.path.exists(test_ds_path):
        shutil.rmtree(test_ds_path)

    dirname_label_df = get_dirname_label_map_db(model_type, station_config_nr)
    dirname_quality_map = get_dirname_quality_map()
    print(dirname_quality_map)

    train_dirname_label_df, val_dirname_label_df, test_dirname_label_df = get_subject_ids(dirname_label_df, baseline_test_subject_ids)

    create_new_baseline_from_df(train_dirname_label_df, baseline_train_dir, dirname_quality_map)
    create_new_baseline_from_df(val_dirname_label_df, baseline_val_dir, dirname_quality_map)
    create_new_baseline_from_df(test_dirname_label_df, test_ds_path, dirname_quality_map)

    print("Done creating baseline data structure")

    # crop all images in the new data structure
    crop_all_images_in_path(baseline_train_dir, model_type)
    crop_all_images_in_path(baseline_val_dir, model_type)
    crop_all_images_in_path(test_ds_path, model_type)

#create_EBUS_baseline()


#------------------ EBUS sequence helper functions ------------------#

def get_dirname_label_map_full_video():
    dirname_label_df = []
    patient_id = 0
    for full_video in full_video_path:
        for patient in os.listdir(full_video):
            patient_path = os.path.join(full_video, patient)
            for sequence in os.listdir(patient_path):
                # check if sequence is not equal to Sequence_001 and throw error if it is
                if sequence != 'Sequence_001':
                    raise ValueError('More than one sequence found for patient: ', patient)
                sequence_path = os.path.join(patient_path, sequence)
                for file in os.listdir(sequence_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(sequence_path, file)
                        df = pd.read_csv(file_path, sep=';')
                        zero_start_index, zero_end_index = select_zero_sequence(df)
                        label_start_indices, label_end_indices, label = get_label_sequences(df)
                        for i in range(len(label_start_indices)):
                            if label[i] not in list(stations_config.keys()):
                                continue  # skip this sequence if label is not in stations_config
                            dirname_label_df.append({'dirname': sequence_path,
                                                     'patient_id': str(patient_id),
                                                     'label': label[i],
                                                     'start_index': label_start_indices[i],
                                                     'end_index': label_end_indices[i]})

                        dirname_label_df.append({'dirname': sequence_path,
                                                 'patient_id': str(patient_id),
                                                 'label': '0',
                                                 'start_index': zero_start_index,
                                                 'end_index': zero_end_index})

            patient_id += 1


    dirname_label_df = pd.DataFrame(dirname_label_df)
    return dirname_label_df

def select_zero_sequence(df):
    random_zero_sequence_start_index = None
    random_zero_sequence_end_index = None
    # find the start and end indexes of all zero sequences in the dataframe
    zero_sequence_start_indexes = df.index[(df['label'] == '0') & (df['label'].shift(1) != '0')].tolist()
    zero_sequence_end_indexes = df.index[(df['label'] == '0') & (df['label'].shift(-1) != '0')].tolist()

    #exclude the first and last indexes
    zero_sequence_start_indexes = zero_sequence_start_indexes[1:-1]
    zero_sequence_end_indexes = zero_sequence_end_indexes[1:-1]

    # pick a random zero sequence from the dataframe and return the start and end indexes
    random_zero_sequence_start_index = random.choice(zero_sequence_start_indexes)
    random_zero_sequence_end_index = zero_sequence_end_indexes[zero_sequence_start_indexes.index(random_zero_sequence_start_index)]

    return random_zero_sequence_start_index, random_zero_sequence_end_index

def get_label_sequences(df):
    # finds start and end indices of all label sequences in the dataframe except where the label is 0,
    # i.e. for the start index: where the label goes from being 0 to something else
    sequence_start_indexes = df.index[(df['label'] != '0') & (df['label'].shift(1) == '0')].tolist()
    sequence_end_indexes = df.index[(df['label'] != '0') & (df['label'].shift(-1) == '0')].tolist()
    # get labels of all sequences based on start index
    sequence_label_list = df.loc[sequence_start_indexes, 'label'].tolist()
    return sequence_start_indexes, sequence_end_indexes, sequence_label_list

def create_new_sequence_from_df(df, sequence_dir, dirname_quality_map):
    # loop through train and val sequence_label_df and copy the images to the new data structure
    for index, row in df.iterrows():
        dirname = row['dirname']
        label = row['label']
        start = row['start_index']
        end = row['end_index']
        # split dirname at '/' and take the last two elements
        dirname_split = dirname.split('/')
        name_split = dirname_split[-3].split('_')
        name = '_'.join(name_split[:-2])
        new_dir = os.path.join(sequence_dir, name + '_'  + dirname_split[-2], label)
        print("new_dir: ", new_dir)

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        # TODO: make sure this part works for new retrieval method
        if filter_data:
            # only copy over images labeled as 'good' or 'ok' quality
            if dirname_quality_map[dirname] != 'poor':
                # copying all files from the source directory to destination directory
                copy_directory_sequence(dirname, new_dir, start, end)
                print("Copied: ", dirname, " to ", new_dir)
        else:
            # copying all files from the source directory to destination directory
            copy_directory_sequence(dirname, new_dir, start, end)
            print("Copied: ", dirname, " to ", new_dir)


def copy_directory_sequence(src_dir, dst_dir, start, end):
    # for each frame in the sequence, copy the frame to the new directory
    for i in range(start, end):
        frame = 'frame_' + str(i) + '.png'
        src = os.path.join(src_dir, frame)
        dst = os.path.join(dst_dir, frame)
        shutil.copyfile(src, dst)


#------------------ create EBUS sequence ------------------#
def create_EBUS_sequence():

    sequence_train_dir = os.path.join(data_path, 'train')
    sequence_val_dir = os.path.join(data_path, 'val')

    # remove old data structure
    if os.path.exists(sequence_train_dir):
        shutil.rmtree(sequence_train_dir)

    if os.path.exists(sequence_val_dir):
        shutil.rmtree(sequence_val_dir)

    if os.path.exists(test_ds_path):
        shutil.rmtree(test_ds_path)

    full_video_dirname_label_df = get_dirname_label_map_full_video()
    dirname_quality_map = get_dirname_quality_map()

    train_dirname_label_df, val_dirname_label_df, test_dirname_label_df = get_subject_ids(full_video_dirname_label_df)

    create_new_sequence_from_df(train_dirname_label_df, sequence_train_dir, dirname_quality_map)
    create_new_sequence_from_df(val_dirname_label_df, sequence_val_dir, dirname_quality_map)
    create_new_sequence_from_df(test_dirname_label_df, test_ds_path, dirname_quality_map)

    print("Done creating sequence data structure")


    # crop all images in the new data structure
    crop_all_images_in_path(sequence_train_dir, model_type)
    crop_all_images_in_path(sequence_val_dir, model_type)
    crop_all_images_in_path(test_ds_path, model_type)

create_EBUS_sequence()


# old code, TODO: remove?
def balance_dataset(root_folder):
    """
    This function is used to balance the dataset by deleting frames from the folders with the most frames

    :param root_folder:
    :return:
    """
    # Step 1: Find the folder with the least number of frames
    min_frames = float('inf')
    for station in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, station)
        if os.path.isdir(folder_path):
            num_frames = len(os.listdir(folder_path))
            if num_frames < min_frames:
                min_frames = num_frames

    # Step 2: Get the list of frames in all folders
    for station in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, station)
        if os.path.isdir(folder_path):
            frame_list = os.listdir(folder_path)

            # Step 3: Delete extra frames from each folder
            if len(frame_list) > min_frames:
                num_frames_to_delete = len(frame_list) - min_frames
                for i in range(num_frames_to_delete):
                    frame_to_delete = frame_list.pop()
                    frame_to_delete_path = os.path.join(folder_path, frame_to_delete)
                    os.remove(frame_to_delete_path)

# balance_dataset(data_path)
