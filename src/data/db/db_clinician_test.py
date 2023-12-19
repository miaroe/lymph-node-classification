import sqlite3
import os
import pandas as pd
from src.resources.config import db_path

def get_clinician_test_df():
    db_name = os.path.join(db_path, "db_blind_classification2.sqlite3")
    table_name = 'imagesequence_clinician_test'
    cnx = sqlite3.connect(db_name)

    df = pd.read_sql_query("SELECT * FROM " + table_name, cnx)
    cnx.close()
    return df


get_clinician_test_df()