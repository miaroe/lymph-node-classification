import sqlite3
import pandas as pd
import shutil
from src.resources.config import *

#https://www.geeksforgeeks.org/python-os-rename-method/
def get_dirname_quality_map():
    # Create db connection
    db_name = os.path.join(db_path, "db_modified.sqlite3")

    if model_type == 'baseline':
        table_name = 'imagesequence_with_quality'
    elif model_type == 'combined_baseline' or model_type == 'combined_sequence':
        table_name = 'imagesequence_with_quality_combined'
    else:
        raise Exception("Invalid model_type")

    cnx = sqlite3.connect(db_name)

    df = pd.read_sql_query("SELECT * FROM " + table_name, cnx)

    # Create a dictionary to store the mappings
    dirname_quality_map = {}

    for index, row in df.iterrows():
        format = row['format']
        quality = row['image_quality']
        dirname = os.path.dirname(format)

        dirname_quality_map[dirname] = quality
    cnx.close()
    return dirname_quality_map


















