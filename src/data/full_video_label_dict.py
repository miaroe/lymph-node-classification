from src.resources.config import *
import os
import pandas as pd


def get_frame_label_dict(full_video_path):
    frame_label_dict = {}
    labels_path = os.path.join(full_video_path, 'labels.csv')
    # read the csv file using pandas
    df = pd.read_csv(labels_path, sep=';')

    # create a dictionary to store the mappings
    for index, row in df.iterrows():
        frame = (row['path']).split('\\')[-1].replace('.png', '')
        frame_label_dict[frame] = row['label']

    return frame_label_dict


#get_frame_label_dict(local_full_video_path)