import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.count_station_distribution import count_station_distribution

def station_distribution_figure_and_report(train_ds, val_ds, test_ds, num_stations, stations_config, reports_path):
    """
    Plot station distribution and save report to csv
    :param train_ds:
    :param val_ds:
    :param num_stations:
    :param stations_config:
    :param reports_path:
    :return:
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16, 10))
    text_kwargs = {'fontsize': 13, 'horizontalalignment': 'center'}
    width = 0.3  # width of bars
    colors = ['#80B1D3', '#8dd3c7', '#bebada']

    for ds_label, batch_gen_i, offset in zip(('train', 'validation', 'test'), (train_ds, val_ds, test_ds), (-width + width/2, width/2, width + width/2)):

        counter = 0 if ds_label == 'train' else 1 if ds_label == 'validation' else 2
        count_array = count_station_distribution(batch_gen_i, num_stations)

        # -------------------------------------------- FIGURE --------------------------------------------
        relative_frequency = 100 * count_array / np.sum(count_array)
        x = np.arange(len(relative_frequency))  # the label locations

        ax.bar(x=x + offset, height=relative_frequency, width=width, label=ds_label, color=colors[counter], align='center')

        for j in x:
            ax.text(x=j + offset, y=relative_frequency[j] + width,
                     s=str(np.round(relative_frequency[j], 1)) + '%', **text_kwargs)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(stations_config.keys(), fontsize=14)
        ax.legend(fontsize=16)

        plt.xlabel('Lymph node station', fontsize=20)
        plt.ylabel('Frequency (%)', fontsize=20)
        ax.set_ylim(0, 32)
        ax.set_xlim(-0.5, len(relative_frequency) - 0.2)

        # Save figure to reports folder
        fig_path = os.path.join(reports_path, 'figures/')
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(fig_path + 'station_distribution.png', bbox_inches='tight')


        # -------------------------------------------- REPORT --------------------------------------------
        df = []
        for idx, elem in enumerate(count_array):
            df.append([list(stations_config.keys())[idx], elem])
        df.append(['Total', np.sum(count_array)])

        df = pd.DataFrame(df, columns=['Station', 'Files'])
        df.to_csv(reports_path + f'station_distribution_{ds_label}.csv', sep='\t', index=False)



