import sqlite3
import pandas as pd

from src.resources.config import *

#TODO: remove this file, not used
def get_dirname_label_map_db(model_type, station_config_nr):
    """
    Create a dictionary mapping dirname to label for the given model_type and station_config_nr
    :param model_type:
    :param station_config_nr:
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