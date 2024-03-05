import sqlite3
import pandas as pd
from src.resources.config import *

def get_dirname_good_quality_frame_map():
    # Create db connection
    db_name = os.path.join(db_path, "db_subseq.sqlite3")
    table_name = "subsequence_quality_combined"

    cnx = sqlite3.connect(db_name)

    df = pd.read_sql_query("SELECT * FROM " + table_name, cnx)

    # Create a dictionary to store the mappings
    dirname_quality_map = []

    for index, row in df.iterrows():
        format = row['format']
        frame_nr = row['frame_nr']
        seq_number = row['seq_number']

        dirname = os.path.dirname(format)
        dirname_split = dirname.split('/')
        name = dirname_split[-3] + '_' + dirname_split[-2]
        label = dirname_split[-1].split('_')[1]
        dirname = os.path.join(name, label)
        dirname_quality_map.append({'dirname': dirname, 'frame_nr': frame_nr, 'seq_number': seq_number})
    cnx.close()
    dirname_quality_map = pd.DataFrame(dirname_quality_map)
    print(dirname_quality_map)
    return dirname_quality_map

get_dirname_good_quality_frame_map()

















