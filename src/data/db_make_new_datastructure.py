import sqlite3
import pandas as pd
import shutil
from src.resources.config import *
from db_image_quality import get_sequence_quality_map


# https://www.geeksforgeeks.org/python-os-rename-method/
def get_sequence_label_map():
    # Create db connection
    db_name = os.path.join(db_path, "db_modified.sqlite3")
    table_name = 'imagesequence_with_quality'
    cnx = sqlite3.connect(db_name)

    df = pd.read_sql_query("SELECT * FROM " + table_name, cnx)

    # Create a dictionary to store the mappings
    sequence_label_map = {}

    for index, row in df.iterrows():
        path = row['format']
        label = row['label']
        dirname = os.path.dirname(path)

        sequence_label_map[dirname] = label
    cnx.close()
    return sequence_label_map


# copy sequence into a new dircetory under the label name
# rename the files to the frame number
def copy_directory(src_dir, dst_dir, label, frame_number_dict):
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


# to make new data structure from EBUS_Levanger directory as source, delete existing
# EBUS_Levanger_new and call this function
def make_new_data_structure():
    frame_number_dict = {'4L': 0,
                         '4R': 0,
                         '7L': 0,
                         '7R': 0,
                         '10L': 0,
                         '10R': 0,
                         '11L': 0,
                         '11R': 0,
                         '7': 0
                         }
    sequence_label_map = get_sequence_label_map()

    for dirname, label in sequence_label_map.items():
        new_dir = os.path.join(data_path, label)  # (data_path + '_new', label)

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            frame_number_dict[label] = 0

        if mask_poor:
            # only copy over images labeled as 'good' or 'ok' quality
            sequence_quality_map = get_sequence_quality_map()
            if sequence_quality_map[dirname] != 'poor':
                # copying all files from the source directory to destination directory
                frame_number_dict = copy_directory(dirname, new_dir, label, frame_number_dict)
                print("Copied: ", dirname, " to ", new_dir)
                print(frame_number_dict)
        else:
            frame_number_dict = copy_directory(dirname, new_dir, label, frame_number_dict)
            print("Copied: ", dirname, " to ", new_dir)
            print(frame_number_dict)

#make_new_data_structure()

def balance_dataset(root_folder):
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

#balance_dataset(data_path)
