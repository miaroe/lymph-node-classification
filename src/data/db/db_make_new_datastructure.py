import sqlite3
import pandas as pd
import shutil
import numpy as np
import cv2

from src.resources.config import *
from src.data.db.db_image_quality import get_dirname_quality_map
from alive_progress import alive_bar

#------------------ Helper functions ------------------#

def get_dirname_label_map(model_type, station_config_nr):
    """
    Create a dictionary mapping dirname to label
    :param model_type:
    :return:
    """
    # Create db connection
    db_name = os.path.join(db_path, "db_modified.sqlite3")
    if model_type == 'baseline':
        table_name = 'imagesequence_with_quality'
    elif model_type == 'combined_baseline':
        table_name = 'imagesequence_with_quality_combined'
    else:
        raise Exception("Invalid model_type")
    cnx = sqlite3.connect(db_name)

    df = pd.read_sql_query("SELECT * FROM " + table_name, cnx)

    # Create a dictionary to store the mappings
    dirname_label_map = []

    for index, row in df.iterrows():
        path = row['format']
        patient = row['subject_id']
        label = row['label']
        if station_config_nr == 4: # without 10L
            if label == '10L' or label == '7':
                continue
        if station_config_nr == 5: # combine 7L and 7R
            if label == '7L' or label == '7R':
                label = '7'
        if station_config_nr == 6: # combine 7L and 7R, without 10L
            if label == '7L' or label == '7R':
                label = '7'
            elif label == '10L':
                continue
        dirname = os.path.dirname(path)

        dirname_label_map.append({'dirname': dirname,
                                   'patient_id': patient,
                                   'label': label})
    cnx.close()
    dirname_label_df = pd.DataFrame(dirname_label_map)
    print('dirname_label_map: ', dirname_label_df)
    return dirname_label_df



def copy_directory(src_dir, dst_dir, label, frame_number_dict):
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


def get_subject_ids(dirname_label_df, baseline_test_subject_ids):
    """
    chooses which patients to use for training and validation at random according to subject_id column in db and validation_split
    does not include test patients

    :param dirname_label_df:
    :param baseline_test_subject_ids:
    :return:
    """

    unique_patient_ids = dirname_label_df['patient_id'].unique()
    unique_patient_ids = [patient_id for patient_id in unique_patient_ids if patient_id not in baseline_test_subject_ids]

    np.random.shuffle(unique_patient_ids)

    num_val_patients = int(len(unique_patient_ids) * validation_split)

    train_patient_ids = unique_patient_ids[num_val_patients:]
    val_patient_ids = unique_patient_ids[:num_val_patients]

    train_dirname_label_df = dirname_label_df[dirname_label_df['patient_id'].isin(train_patient_ids)]
    val_dirname_label_df = dirname_label_df[dirname_label_df['patient_id'].isin(val_patient_ids)]
    test_dirname_label_df = dirname_label_df[dirname_label_df['patient_id'].isin(baseline_test_subject_ids)]

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # save the train, val and test sequence_label_df to csv
    train_dirname_label_df.to_csv(os.path.join(data_path, 'train_dirname_label_df.csv'))
    val_dirname_label_df.to_csv(os.path.join(data_path, 'val_dirname_label_df.csv'))
    test_dirname_label_df.to_csv(os.path.join(data_path, 'test_dirname_label_df.csv'))

    return train_dirname_label_df, val_dirname_label_df, test_dirname_label_df

def create_new_datastructure_from_df(df, baseline_dir, dirname_quality_map, station_config_nr):
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
                frame_number_dict = copy_directory(dirname, new_dir, label, frame_number_dict)
                print("Copied: ", dirname, " to ", new_dir)
                print("frame_number_dict: ", frame_number_dict)

        else:
            # copying all files from the source directory to destination directory
            frame_number_dict = copy_directory(dirname, new_dir, label, frame_number_dict)
            print("Copied: ", dirname, " to ", new_dir)
            print("frame_number_dict: ", frame_number_dict)

def crop_all_images_in_path(path):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                image = image[100:1035, 530:1658]
                cv2.imwrite(image_path, image)
                print("Cropped: ", image_path)



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

    dirname_label_df = get_dirname_label_map(model_type, station_config_nr)
    dirname_quality_map = get_dirname_quality_map()
    print(dirname_quality_map)

    train_dirname_label_df, val_dirname_label_df, test_dirname_label_df = get_subject_ids(dirname_label_df, baseline_test_subject_ids)

    create_new_datastructure_from_df(train_dirname_label_df, baseline_train_dir, dirname_quality_map, station_config_nr)
    create_new_datastructure_from_df(val_dirname_label_df, baseline_val_dir, dirname_quality_map, station_config_nr)
    create_new_datastructure_from_df(test_dirname_label_df, test_ds_path, dirname_quality_map, station_config_nr)

    print("Done creating baseline data structure")

    # crop all images in the new data structure
    crop_all_images_in_path(baseline_train_dir)
    crop_all_images_in_path(baseline_val_dir)
    crop_all_images_in_path(test_ds_path)

create_EBUS_baseline()

# remember to run make_EBUS_Levanger_baseline first to include all data, if not some will have been transferred to test set
# not used? TODO: remove
def make_EBUS_combined_baseline():
    Levanger_baseline_dir = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger_baseline'
    StOlavs_baseline_dir = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_StOlavs_baseline'
    new_dir = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_combined_baseline'

    # remove old data structure
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)

    # for each folder in Levanger and StOlavs, copy all files to new_dir with the same folder name, but filename is set according to index
    frame_number_dict = {'4L': 0,
                         '4R': 0,
                         '7L': 0,
                         '7R': 0,
                         '10L': 0,
                         '10R': 0,
                         '11L': 0,
                         '11R': 0
                         }
    for dirname in os.listdir(Levanger_baseline_dir):
        src_dir = os.path.join(Levanger_baseline_dir, dirname)
        dst_dir = os.path.join(new_dir, dirname)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            frame_number_dict[dirname] = 0
        frame_number_dict = copy_directory(src_dir, dst_dir, dirname, frame_number_dict)
        print("Copied: ", src_dir, " to ", dst_dir)
        print(frame_number_dict)

    for dirname in os.listdir(StOlavs_baseline_dir):
        src_dir = os.path.join(StOlavs_baseline_dir, dirname)
        dst_dir = os.path.join(new_dir, dirname)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            frame_number_dict[dirname] = 0
        frame_number_dict = copy_directory(src_dir, dst_dir, dirname, frame_number_dict)
        print("Copied: ", src_dir, " to ", dst_dir)
        print(frame_number_dict)

#make_EBUS_combined_baseline()

#------------------ Create EBUS sequence ------------------#

def make_EBUS_Levanger_sequence(mask_poor=False):
    """
    copy all images from the source directory to destination directory data_path
    if mask poor is true, only copy over images labeled as 'good' or 'ok' quality

    :param mask_poor:
    :return:
    """
    data_path_standard = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger'

    # add alive progress bar
    with alive_bar(len(os.listdir(data_path_standard)), title='new directory', bar='bubbles', spinner='fishes') as bar:
        for dirname in os.listdir(data_path_standard):
            print(dirname)
            new_dir = os.path.join(data_path, dirname)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            for filename in os.listdir(os.path.join(data_path_standard, dirname)):
                print(filename)
                filename_new = filename.split('_')[1]
                src = os.path.join(data_path_standard, dirname, filename)
                dst = os.path.join(new_dir, filename_new)
                if mask_poor:
                    # only copy over images labeled as 'good' or 'ok' quality
                    dirname_quality_map = get_dirname_quality_map()
                    if dirname_quality_map[src] == 'poor':
                        print("mask poor")
                        continue
                if not os.path.exists(dst):
                    os.makedirs(dst)
                for file in os.listdir(src):
                    print('copying file: ', file)
                    shutil.copyfile(os.path.join(src, file), os.path.join(dst, file))
            bar()


# make_EBUS_Levanger_sequence(mask_poor=True)


# not used? TODO: remove
def make_EBUS_StOlavs_baseline():
    dir = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_StOlavs'
    new_dir = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_StOlavs_baseline'


    frame_number_dict = {'4L': 0,
                         '4R': 0,
                         '7L': 0,
                         '7R': 0,
                         '10L': 0,
                         '10R': 0,
                         '11L': 0,
                         '11R': 0
                         }
    for dirname in os.listdir(dir):
        for filename in os.listdir(os.path.join(dir, dirname)):
            label = filename.split('_')[1]
            src = os.path.join(dir, dirname, filename)
            dst = os.path.join(new_dir, label)

            if not os.path.exists(dst):
                os.makedirs(dst)
                frame_number_dict[label] = 0

            frame_number_dict = copy_directory(src, dst, label, frame_number_dict)
            print("Copied: ", src, " to ", dst)
            print(frame_number_dict)


#make_EBUS_StOlavs_baseline()



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
