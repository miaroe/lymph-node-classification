import sqlite3
import pandas as pd
from matplotlib import pyplot as plt

from src.resources.config import *

def plot_nr_of_frames():
    plt.style.use('classic')  # Using a seaborn style for a more modern look
    db_name = os.path.join(db_path, "db_modified.sqlite3")
    table_name = 'imagesequence_with_quality'
    cnx = sqlite3.connect(db_name)


    df = pd.read_sql_query("SELECT * FROM " + table_name, cnx)

    # Filter out station '7'
    df = df[df['label'] != '7']

    # Define the correct order of the labels
    label_order = ['4L', '4R', '7L', '7R', '10L', '10R', '11L', '11R']
    df['label'] = pd.Categorical(df['label'], categories=label_order, ordered=True)
    df = df.sort_values('label')

    # Updated color map with more aesthetic choices
    colors = {
        '4L': '#3498db',  # Blue
        '4R': '#2ecc71',  # Green
        '7L': '#e74c3c',  # Coral
        '7R': '#8e44ad',  # Cyan
        '10L': '#fd79a8',  # Pink
        '10R': '#f1c40f',  # Yellow
        '11L': '#95a5a6',  # Grey
        '11R': '#f39c12'  # Golden
    }

    # Plot number of frames as dots in different colors based on the label map
    for label, group in df.groupby('label'):
        plt.scatter(group['nr_of_frames'], [label] * len(group), color=colors[label], s=50)  # Adjust marker size

    # Calculate overall mean of number of frames
    mean_total = df['nr_of_frames'].mean()
    mean = df.groupby('label')['nr_of_frames'].mean()

    plt.scatter(mean, mean.index, marker='X', color='black', s=50, label='Mean per station')

    # Adding mean annotation for better visibility
    plt.axvline(x=mean_total, color='black', linestyle='--', linewidth=1)
    plt.text(mean_total, -1, f'Total Mean: {mean_total:.2f}', verticalalignment='bottom',
             horizontalalignment='left', color='black', fontsize=12)

    plt.xlabel('Number of Frames')
    plt.ylabel('Station')
    plt.title('Number of Frames in Each Station Video')
    plt.xlim(0, df['nr_of_frames'].max() + 100)  # Set x-axis to start at 0

    plt.legend()
    plt.show()
    cnx.close()

plot_nr_of_frames()

