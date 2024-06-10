import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

from src.resources.config import get_stations_config


subject_ids = [
    709, 710, 711, 712, 713, 714, 715, 716, 717, 718,
    719, 720, 721, 722, 723, 724, 725, 726, 727, 728,
    729, 730, 731, 732, 733, 734, 735, 736, 737, 738,
    739, 740, 741, 742, 743, 744, 745, 746, 747, 748,
    799, 800, 807, 808, 809, 810, 811, 812, 813, 814,
    815, 816, 818
]

row_counts = [
    640, 174, 290, 33, 247, 230, 429, 338, 197, 57,
    87, 250, 128, 233, 175, 325, 62, 284, 296, 479,
    362, 408, 220, 309, 270, 373, 434, 385, 351, 207,
    213, 341, 347, 250, 301, 167, 285, 215, 265, 255,
    88, 214, 407, 254, 376, 304, 358, 471, 413, 494,
    223, 310, 337
]

total_nr_of_frames = [ # total: 27 074
    903, 359, 398, 66, 278, 251, 489, 520, 442, 242,
    587, 366, 797, 341, 414, 401, 75, 679, 395, 637,
    865, 654, 486, 491, 710, 764, 812, 607, 558, 600,
    338, 616, 768, 622, 477, 281, 546, 384, 462, 315,
    416, 552, 753, 385, 710, 400, 652, 602, 494, 749,
    423, 483, 459
]

'''
total_nr_of_frames = [
    910, 308, 405, 68, 284, 257, 497, 528, 383, 115,
    278, 371, 267, 347, 282, 408, 77, 544, 344, 608,
    418, 533, 407, 410, 600, 492, 819, 582, 458, 609,
    300, 623, 680, 463, 432, 261, 530, 338, 397, 318,
    240, 264, 645, 391, 603, 369, 502, 610, 455, 642,
    401, 373
]
'''
# total nr of frames: 22,446
# total number of stations: 307
# rejected stations: 385 - 307 = 78

def get_total_nr_of_frames():

    path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence_segmentation/Levanger_and_StOlavs'
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for patient in os.listdir(folder_path):
            patient_nr_of_frames = 0
            patient_path = os.path.join(folder_path, patient)
            for station in os.listdir(patient_path):
                station_path = os.path.join(patient_path, station)
                frame_names = os.listdir(station_path)
                patient_nr_of_frames += len(frame_names)
            print(f'Patient: {patient}, Total number of frames: {patient_nr_of_frames}')

#get_total_nr_of_frames()


def get_subject_data():
    subject_data = {
        'Subject_ID': subject_ids,
        'Row_Count': row_counts,
        'Total_Frames': total_nr_of_frames
    }
    df_subjects = pd.DataFrame(subject_data)
    return df_subjects


def plot_subject_quality():
    plt.style.use('classic')
    df_subjects = get_subject_data()

    # Calculate percentage of good quality frames
    df_subjects['Percentage'] = df_subjects['Row_Count'] / df_subjects['Total_Frames'] * 100

    # Sort the dataframe by percentage
    df_subjects = df_subjects.sort_values(by='Percentage')

    # Create an x-axis with numbers from 1 onwards
    x_axis = np.arange(1, len(df_subjects) + 1)

    # Plotting the percentage of good quality frames
    plt.figure(figsize=(12, 7), facecolor='white')

    # Scatter plot
    plt.scatter(x_axis, df_subjects['Percentage'], color='#8DD3C7', marker='o', s=100)

    # Adding labels and title
    plt.xlabel('Patients', fontsize=16)
    plt.ylabel('Good Quality Frames (%)', fontsize=16)

    # Adding xticks with empty labels
    plt.xticks(x_axis, [''] * len(x_axis))

    # Formatting y-axis labels with percentage sign
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    # Adjusting x-axis limits
    plt.xlim(1, len(df_subjects))

    # Removing spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(top=False, right=False)

    # Removing gridlines
    plt.grid(False)

    # Tight layout
    plt.tight_layout()

    # Saving the plot
    fig_path = '/home/miaroe/workspace/lymph-node-classification/figures/'
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + 'subject_quality_percentage_scatter_plot.png', bbox_inches='tight')

#plot_subject_quality()




def plot_subject_quality_combined():
    plt.style.use('classic')
    df_subjects = get_subject_data()

    # Calculate percentage of good quality frames
    df_subjects['Percentage'] = df_subjects['Row_Count'] / df_subjects['Total_Frames'] * 100

    # Plotting the violin plot with a box plot overlay
    plt.figure(figsize=(10, 6), facecolor='white')

    # Create a violin plot
    ax = sns.violinplot(y=df_subjects['Percentage'], color='#8DD3C7', inner=None, saturation=0.8, cut=0)
    # set opacity of the violin plot
    for patch in ax.collections:
        patch.set_edgecolor('#8DD3C7')
        patch.set_alpha(0.3)

    # Overlay the box plot
    sns.boxplot(y=df_subjects['Percentage'], width=0.1, boxprops={'zorder': 2}, ax=ax, color='#8DD3C7', whis=1.5)

    # Adding labels and title
    plt.ylabel('Good Quality Frames', fontsize=16)
    plt.xlabel('Patients', fontsize=16)


    plt.ylim(0, 100)

    # Formatting y-axis labels with percentage sign
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    # Removing spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(top=False, right=False, bottom=False)

    # Removing gridlines
    plt.grid(False)

    # Tight layout
    plt.tight_layout()

    # Saving the plot
    fig_path = '/home/miaroe/workspace/lymph-node-classification/figures/'
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + 'subject_quality_combined_plot.png', bbox_inches='tight')
    plt.show()

plot_subject_quality_combined()


# Calculate percentage of good quality frames
percentage = [row / total * 100 for row, total in zip(row_counts, total_nr_of_frames)]

# Create a DataFrame
df_subjects = pd.DataFrame({
    'Subject_ID': subject_ids,
    'Row_Count': row_counts,
    'Total_Frames': total_nr_of_frames,
    'Percentage': percentage
})
# Calculate Q1, Q3, and IQR
Q1 = df_subjects['Percentage'].quantile(0.25)
Q3 = df_subjects['Percentage'].quantile(0.75)
IQR = Q3 - Q1

# Calculate lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df_subjects[(df_subjects['Percentage'] < lower_bound) | (df_subjects['Percentage'] > upper_bound)]

# Print the results
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")
print("Outliers:")
print(outliers)


def plot_quality():
    plt.style.use('classic')
    colors = ['#80B1D3', '#8dd3c7', '#bebada']
    class_data = {
        'Class': get_stations_config(3).keys(),
        'Good_Quality_Frames': [2920, 2420, 2054, 1916, 515, 1679, 1257, 2402], #15163
        'Total_Frames': [4684, 4447, 3306, 3398, 878, 3858, 2751, 3752] #27074 0,5623845756%
    }
    df_classes = pd.DataFrame(class_data)

    # Plotting a bar plot
    plt.figure(figsize=(12, 7), facecolor='white')
    bar_width = 0.4  # Width of the bars

    # Setting positions of bars
    r1 = np.arange(len(df_classes['Class']))  # positions for the first set of bars
    r2 = [x + bar_width for x in r1]  # positions for the second set of bars

    # Creating bars
    plt.bar(r1, df_classes['Good_Quality_Frames'], color=colors[2], width=bar_width, label='Good quality')
    plt.bar(r2, df_classes['Total_Frames'], color=colors[0], width=bar_width, label='Total')

    # Adding xticks
    plt.xlabel('Lymph node station', fontsize=16)
    plt.xticks([r + bar_width/2 for r in range(len(df_classes['Class']))], df_classes['Class'])

    # Adjusting x-axis limits
    plt.xlim(-0.5, len(df_classes['Class']))

    # Adding labels and title
    plt.ylabel('Number of frames', fontsize=16)

    # remove spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(top=False, right=False)

    # Creating legend & showing the plot
    plt.legend(fontsize=16)

    fig_path = '/home/miaroe/workspace/lymph-node-classification/figures/'
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + 'quality_bar_plot.png', bbox_inches='tight')


