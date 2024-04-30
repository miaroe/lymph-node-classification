import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.resources.config import get_stations_config

plt.style.use('classic')
colors = ['#80B1D3', '#8dd3c7', '#bebada']
class_data = {
    'Class': get_stations_config(3).keys(),
    'Good_Quality_Frames': [2971, 2420, 2054, 1916, 515, 1691, 1257, 2402], #15226
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
