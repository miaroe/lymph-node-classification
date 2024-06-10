import shutil

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from src.data.make_new_datastructure import get_dirname_label_map_full_video
from src.resources.config import *
from src.data.make_new_datastructure import crop_all_images_in_path


def get_dirname_label_map_sequence_cv():
    dirname_label_df = []
    patient_id = 0
    path = os.path.join(data_path, 'data')
    for patient in os.listdir(path):
        patient_path = os.path.join(path, patient)
        for station in os.listdir(patient_path):
            station_path = os.path.join(patient_path, station)
            dirname_label_df.append({'dirname': station_path,
                                     'patient_id': str(patient_id),
                                     'label': station})

        patient_id += 1
    dirname_label_df = pd.DataFrame(dirname_label_df)
    return dirname_label_df



def create_k_fold_cv_data_structure(n_splits, random_state):

    #kfold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    dirname_label_df = get_dirname_label_map_sequence_cv()

    unique_patient_ids = dirname_label_df['patient_id'].unique()
    print('unique_patient_ids: ', unique_patient_ids)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(unique_patient_ids)):
        train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=None)

        #save each fold to a csv file using the patient_id to extract info from dirname_label_df
        train_df = dirname_label_df[dirname_label_df['patient_id'].isin(unique_patient_ids[train_ids])]
        val_df = dirname_label_df[dirname_label_df['patient_id'].isin(unique_patient_ids[val_ids])]
        test_df = dirname_label_df[dirname_label_df['patient_id'].isin(unique_patient_ids[test_ids])]

        fold_path = os.path.join(data_path, f'fold_{fold}_v2')
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        train_df.to_csv(os.path.join(fold_path, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_path, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(fold_path, 'test.csv'), index=False)

        print(f'Fold {fold} created')

def copy_station_cv(src_dir, dst_dir, label, start, end, frame_number_dict):
    """
    copy sequence into a new dircetory under the label name

    :param src_dir: source directory
    :param dst_dir: destination directory
    :param label: label
    :param start: start index
    :param end: end index
    :param frame_number_dict: dictionary to keep track of frame number
    """

    if os.path.isdir(src_dir):
        # rename file names and copy to new directory
        for i in range(start, end + 1):
            frame = 'frame_' + str(i) + '.png'
            src = os.path.join(src_dir, frame)
            dst = os.path.join(dst_dir, frame)

            shutil.copyfile(src, dst)
            frame_number_dict[label] += 1
        return frame_number_dict
    else:
        raise Exception("Directory does not exist")

def create_new_datastructure_from_df_cv(dir):
    """
    creates a new data structure
    :param dir: directory
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
    df = get_dirname_label_map_full_video()

    for index, row in df.iterrows():
        dirname = row['dirname']
        label = row['label']
        start = row['start_index']
        end = row['end_index']

        dirname_split = dirname.split('/')
        name_split = dirname_split[-3].split('_')
        name = '_'.join(name_split[:-2])
        dst_dirname = os.path.join(name + '_' + dirname_split[-2], label)  # hospital_Patient_nr/label
        new_dir = os.path.join(dir, dst_dirname)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            frame_number_dict[label] = 0

        # copying all files from the source directory to destination directory
        frame_number_dict = copy_station_cv(dirname, new_dir, label, start, end, frame_number_dict)
        print("Copied: ", dirname, " to ", new_dir)
        print("frame_number_dict: ", frame_number_dict)

    print("Done creating new data structure")

    # crop all images in the new data structure
    crop_all_images_in_path(dir, model_type)


    n_splits = 5
    create_k_fold_cv_data_structure(n_splits, random_state=42)

#create_new_datastructure_from_df_cv(os.path.join(data_path, 'data'))