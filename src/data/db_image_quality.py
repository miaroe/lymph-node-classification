import sqlite3
import pandas as pd
import shutil
from src.resources.config import *

#https://www.geeksforgeeks.org/python-os-rename-method/
def get_sequence_quality_map():
    # Create db connection
    db_name = os.path.join(db_path, "db_modified.sqlite3")
    table_name = 'imagesequence_with_quality'
    cnx = sqlite3.connect(db_name)

    df = pd.read_sql_query("SELECT * FROM " + table_name, cnx)

    # Create a dictionary to store the mappings
    sequence_quality_map = {}

    for index, row in df.iterrows():
        format = row['format']
        quality = row['image_quality']
        dirname = os.path.dirname(format)

        sequence_quality_map[dirname] = quality
    cnx.close()
    return sequence_quality_map

# not used TODO: delete ?
def delete_directory(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    else:
        raise Exception("Directory does not exist")

def make_new_data_structure():
    sequence_quality_map = get_sequence_quality_map()
    for key, value in sequence_quality_map.items():
        if value == 'poor':
            delete_directory(key)
            print("Deleted: ", key)
        else:
            print("Kept: ", key)