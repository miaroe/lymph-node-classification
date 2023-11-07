import sqlite3
import pandas as pd
from matplotlib import pyplot as plt

from src.resources.config import *

def plot_nr_of_frames():
    plt.style.use('ggplot')
    # Create db connection
    db_name = os.path.join(db_path, "db_modified.sqlite3")
    table_name = 'imagesequence_with_quality'
    cnx = sqlite3.connect(db_name)

    df = pd.read_sql_query("SELECT * FROM " + table_name, cnx)

    for index, row in df.iterrows():
        frames = row['nr_of_frames']
        label = row['label']

        # create color map based on label in different pastel colors
        colors = {'4L': 'lightblue',
                  '4R': 'lightgreen',
                  '7L': 'lightcoral',
                  '7R': 'lightcyan',
                  '10L': 'lightpink',
                  '10R': 'lightyellow',
                  '11L': 'lightgrey',
                  '11R': 'lightgoldenrodyellow',
                  '7': 'lightcoral'
                  }

        # plot nr of frames as dots in different colors based on label map
        plt.scatter(frames, label, color=colors[label])
        plt.xlabel('Number of frames')
        plt.ylabel('Station')
        plt.title('Number of frames in each station video')

    # calculate over all mean of nr of frames
    mean_total = df['nr_of_frames'].mean()
    mean = df.groupby('label')['nr_of_frames'].mean()

    plt.scatter(mean, mean.index, marker='x', label='Mean per station')
    plt.axvline(mean_total, linestyle='dashed', linewidth=2, label='Total mean: ' + str(round(mean_total, 2)))

    plt.legend()

    plt.show()
    cnx.close()


plot_nr_of_frames()
