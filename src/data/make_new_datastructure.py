import shutil
import numpy as np
import cv2
import pandas as pd
import random

from src.resources.config import *
from src.data.db.db_image_quality import get_dirname_good_quality_frame_map
from alive_progress import alive_bar


#------------------ Full video helper functions ------------------#

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
                        label_start_indices, label_end_indices, label, seq_number = get_label_sequences(df)
                        for i in range(len(label_start_indices)):
                            if label[i] not in list(stations_config.keys()):
                                continue  # skip this sequence if label is not in stations_config
                            dirname_label_df.append({'dirname': sequence_path,
                                                     'patient_id': str(patient_id),
                                                     'label': label[i],
                                                     'start_index': label_start_indices[i],
                                                     'end_index': label_end_indices[i],
                                                     'seq_number': seq_number[i]})

            patient_id += 1


    dirname_label_df = pd.DataFrame(dirname_label_df)
    return dirname_label_df


def select_zero_sequence(df):
    # find the start and end indexes of all zero sequences in the dataframe
    zero_sequence_start_indexes = df.index[(df['label'] == '0') & (df['label'].shift(1) != '0')].tolist()
    zero_sequence_end_indexes = df.index[(df['label'] == '0') & (df['label'].shift(-1) != '0')].tolist()

    #exclude the first and last indexes
    zero_sequence_start_indexes = zero_sequence_start_indexes[1:-1]
    zero_sequence_end_indexes = zero_sequence_end_indexes[1:-1]

    # exclude the sequence between 4R and 10R
    for i in range(len(zero_sequence_start_indexes) - 1):
        # check if the label before the start index is 4R and the label after the end index is 10R and exclude the sequence
        if df['label'][zero_sequence_start_indexes[i] - 1] == '4R' and df['label'][zero_sequence_end_indexes[i] + 1] == '10R':
            zero_sequence_start_indexes.pop(i)
            zero_sequence_end_indexes.pop(i)

    # pick a random zero sequence from the dataframe and return the start and end indexes
    random_zero_sequence_start_index = random.choice(zero_sequence_start_indexes)
    random_zero_sequence_end_index = zero_sequence_end_indexes[zero_sequence_start_indexes.index(random_zero_sequence_start_index)]

    # cut the sequence in half and return the first half as the zero sequence
    random_zero_sequence_end_index = random_zero_sequence_start_index + (random_zero_sequence_end_index - random_zero_sequence_start_index) // 2

    return random_zero_sequence_start_index, random_zero_sequence_end_index

def get_seq_number(sequence_label_list):
    label_count = {}
    seq_number = []
    for label in sequence_label_list:
        if label in label_count: label_count[label] += 1
        else: label_count[label] = 1
        seq_number.append(label_count[label])
    return seq_number

def get_label_sequences(df):
    # finds start and end indices of all label sequences in the dataframe except where the label is 0,
    # i.e. for the start index: where the label goes from being 0 to something else
    sequence_start_indexes = df.index[(df['label'] != '0') & (df['label'].shift(1) == '0')].tolist()
    sequence_end_indexes = df.index[(df['label'] != '0') & (df['label'].shift(-1) == '0')].tolist()
    # get labels of all sequences based on start index
    sequence_label_list = df.loc[sequence_start_indexes, 'label'].tolist()
    # check if some labels have multiple sequences and return a list of seq_number
    seq_number = get_seq_number(sequence_label_list)

    # add 0 sequence
    zero_start_index, zero_end_index = select_zero_sequence(df)
    sequence_start_indexes.append(zero_start_index)
    sequence_end_indexes.append(zero_end_index)
    sequence_label_list.append('0')
    seq_number.append(1)
    return sequence_start_indexes, sequence_end_indexes, sequence_label_list, seq_number

#------------------ New data structure helper functions ------------------#

def get_subject_ids(dirname_label_df, val_patient_ids=None, test_patient_ids=None):
    """
    chooses which patients to use for training and validation at random according to subject_id column and validation_split
    if test_patient_ids is not None, then the test_patient_ids are used for testing and the rest for training and validation
    else, the test_patient_ids are chosen at random according to test_split. Same for val_patient_ids.

    :param dirname_label_df:
    :param val_patient_ids:
    :param test_patient_ids:
    :return:
    """

    unique_patient_ids = dirname_label_df['patient_id'].unique()
    print('unique_patient_ids: ', unique_patient_ids)
    print('test_patient_ids: ', test_patient_ids)
    print('val_patient_ids: ', val_patient_ids)
    np.random.shuffle(unique_patient_ids)
    num_val_patients = int(len(unique_patient_ids) * validation_split)
    print('num_val_patients: ', num_val_patients)

    if test_patient_ids is not None: # set test patient ids manually
        unique_patient_ids = [patient_id for patient_id in unique_patient_ids if patient_id not in test_patient_ids]
        if val_patient_ids is not None:
            train_patient_ids = [patient_id for patient_id in unique_patient_ids if patient_id not in val_patient_ids]
        else:
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
            print('train_labels: ', train_labels)
            print('val_labels: ', val_labels)
            print('test_labels: ', test_labels)
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


def copy_good_quality_frames(dirname_quality_map, new_dir, dst_dirname, start, src_dir, dst_dir, struct_type, frame_number_dict, label, seq_number):
    # get all rows in dirname_quality_map where dirname == dst_dirname and seq_number == seq_number
    current_dirname_df = dirname_quality_map[(dirname_quality_map['dirname'] == dst_dirname) & (dirname_quality_map['seq_number'] == seq_number)]
    if not current_dirname_df.empty:
        # create new directory if it doesn't exist
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            frame_number_dict[label] = 0
        # loop through current_dirname_df get frame_nr and copy over the image
        for index, row in current_dirname_df.iterrows():
            frame_nr = row['frame_nr'] + start
            frame = 'frame_' + str(frame_nr) + '.png'
            src = os.path.join(src_dir, frame)
            if struct_type == 'baseline': dst = os.path.join(dst_dir, str(frame_number_dict[label]) + '.png')
            else: dst = os.path.join(dst_dir, frame)
            shutil.copyfile(src, dst)
            frame_number_dict[label] += 1
    return frame_number_dict

def good_quality_frame(dirname_quality_map, dst_dirname, seq_number, i, start):
    # check if the frame is of good quality by checking if the frame_nr + start is in dirname_quality_map
    return not dirname_quality_map[(dirname_quality_map['dirname'] == dst_dirname) & (dirname_quality_map['seq_number'] == seq_number) & (dirname_quality_map['frame_nr'] + start == i)].empty

def copy_station(src_dir, dst_dir, label, start, end, struct_type, frame_number_dict, dirname_quality_map, dirname_good_quality_df, dst_dirname, seq_number):
    """
    copy sequence into a new dircetory under the label name
    rename the files to the frame number and update the frame_number_dict
    create dirname_good_quality_df to keep track of the good quality frames

    :param src_dir: source directory
    :param dst_dir: destination directory
    :param label: label
    :param start: start index
    :param end: end index
    :param struct_type: structure type
    :param frame_number_dict: dictionary to keep track of frame number
    :param dirname_quality_map: dataframe with the quality of the frames
    :param dirname_good_quality_df: dataframe with the good quality frames
    :param dst_dirname: destination directory name
    :param seq_number: sequence number
    :return: updated frame_number_dict
    """

    if os.path.isdir(src_dir):
        # rename file names and copy to new directory
        for i in range(start, end):
            frame = 'frame_' + str(i) + '.png'
            src = os.path.join(src_dir, frame)

            if struct_type == 'baseline':
                dst = os.path.join(dst_dir, str(frame_number_dict[label]) + '.png')
            else: # sequence
                dst = os.path.join(dst_dir, frame)

            if good_quality_frame(dirname_quality_map, dst_dirname, seq_number, i, start): dirname_good_quality_df = dirname_good_quality_df.append({'dirname': dst, 'name': dst_dirname, 'src_frame_nr': i, 'good_quality': 1}, ignore_index=True)
            else: dirname_good_quality_df = dirname_good_quality_df.append({'dirname': dst, 'name': dst_dirname, 'src_frame_nr': int(i), 'good_quality': 0}, ignore_index=True)
            shutil.copyfile(src, dst)
            frame_number_dict[label] += 1
        return frame_number_dict, dirname_good_quality_df
    else:
        raise Exception("Directory does not exist")

def save_dirname_good_quality_df(dirname_good_quality_df, dir):
    dir_split = dir.split('/')
    dir = '/'.join(dir_split[:-1])
    dirname_good_quality_df.to_csv(os.path.join(dir, dir_split[-1] + '_dirname_good_quality_df.csv'))


def create_new_datastructure_from_df(struct_type, df, dir, dirname_quality_map, test=False):
    """
    creates a new data structure for struct_type
    :param df: dataframe
    :param dir: directory
    :param dirname_quality_map: dataframe
    :param test: boolean
    :return: None
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
    dirname_good_quality_df = pd.DataFrame(columns=['dirname', 'name', 'src_frame_nr', 'good_quality'])

    for index, row in df.iterrows():
        dirname = row['dirname']
        label = row['label']
        start = row['start_index']
        end = row['end_index']
        seq_number = row['seq_number']

        dirname_split = dirname.split('/')
        name_split = dirname_split[-3].split('_')
        name = '_'.join(name_split[:-2])
        dst_dirname = os.path.join(name + '_' + dirname_split[-2], label)  # hospital_Patient_nr/label

        if struct_type == 'baseline': new_dir = os.path.join(dir, label)
        else: new_dir = os.path.join(dir, dst_dirname)

        if filter_data and not test: # only copy good quality frames
            frame_number_dict = copy_good_quality_frames(dirname_quality_map, new_dir, dst_dirname, start, dirname, new_dir, struct_type, frame_number_dict, label, seq_number)
            print("Copied: ", dirname, " to ", new_dir)
            print("frame_number_dict: ", frame_number_dict)
        else: # copy all frames
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                frame_number_dict[label] = 0
            # copying all files from the source directory to destination directory
            frame_number_dict, dirname_good_quality_df = copy_station(dirname, new_dir, label, start, end, struct_type, frame_number_dict, dirname_quality_map, dirname_good_quality_df, dst_dirname, seq_number)
            print("Copied: ", dirname, " to ", new_dir)
            print("frame_number_dict: ", frame_number_dict)
    save_dirname_good_quality_df(dirname_good_quality_df, dir)



# ------------------- Crop images -------------------#

def crop_image(filename, folder_path):
    print("Cropping: ", filename)
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        image = image[100:1035, 530:1658]
        cv2.imwrite(image_path, image)
        print("Cropped: ", image_path)


def crop_all_images_in_path(path, struct_type):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if struct_type == 'baseline':
            for filename in os.listdir(folder_path):
                crop_image(filename, folder_path)
        else:
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                for filename in os.listdir(subfolder_path):
                    crop_image(filename, subfolder_path)




#------------------ Create EBUS data ------------------#

def create_EBUS_data(struct_type):
    """
    to make new data structure from FullVideos
    - struct_type decides if the data structure is created for sequence or single image model
    - First the subject ids to use for training, validation and test are retrieved. Test patients can be manually decided (get_subject_ids)
    - Then the new data structure is created from the train, val and test sequence_label_df (create_new_datastructure_from_df)
        - if filter_data is True, only good quality frames are copied to the new data structure
        - if not, all frames are copied and the good quality frames are stored in a dataframe
    - At the end all images in the new data structure are cropped (crop_all_images_in_path)
    model_type, station_config_nr, filter_data, data_path and validation_split, full_video_path are defined in config.py

    :param struct_type: type of data structure to create
    :return: None
    """
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')

    #'Patient_001', 'Patient_022', 'Patient_20231107-093258', 'Patient_20231129-091812'
    test_subject_ids = ['8', '34', '14', '48', '42']
    #'Patient_030', 'Patient_028', 'Patient_021', 'Patient_015', 'Patient_040', 'Patient_034', 'Patient_023', 'Patient_029', 'Patient_20240116-101109', 'Patient_002'
    val_subject_ids = ['7', '11', '16', '21', '22', '26', '31', '38', '49', '53']

    # remove old data structure
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    dirname_label_df = get_dirname_label_map_full_video()
    dirname_quality_map = get_dirname_good_quality_frame_map()

    train_dirname_label_df, val_dirname_label_df, test_dirname_label_df = get_subject_ids(dirname_label_df, val_subject_ids, test_subject_ids)

    create_new_datastructure_from_df(struct_type, train_dirname_label_df, train_dir, dirname_quality_map)
    create_new_datastructure_from_df(struct_type, val_dirname_label_df, val_dir, dirname_quality_map)
    create_new_datastructure_from_df(struct_type, test_dirname_label_df, test_dir, dirname_quality_map, test=True)

    print("Done creating new data structure")

    # crop all images in the new data structure
    crop_all_images_in_path(train_dir, struct_type)
    crop_all_images_in_path(val_dir, struct_type)
    crop_all_images_in_path(test_dir, struct_type)

create_EBUS_data('sequence')

