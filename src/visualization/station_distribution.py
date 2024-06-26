import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.count_station_distribution import count_station_distribution

def station_distribution_figure_and_report(train_ds, val_ds, num_stations, stations_config, reports_path, test_ds=None):
    """
    Plot station distribution and save report to csv
    :param train_ds:
    :param val_ds:
    :param num_stations:
    :param stations_config:
    :param reports_path:
    :return:
    """
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(16, 10))
    text_kwargs = {'fontsize': 13, 'horizontalalignment': 'center'}
    '''
    # separate counts for train val and test
    
    colors = ['#80B1D3', '#8dd3c7', '#bebada']
    width = 0.3  # width of bars
    for ds_label, batch_gen_i, offset in zip(('train', 'validation', 'test'), (train_ds, val_ds, test_ds),
                                             (-width + width / 2, width / 2, width + width / 2)):

        counter = 0 if ds_label == 'train' else 1 if ds_label == 'validation' else 2
        count_array = count_station_distribution(batch_gen_i, num_stations)

        # -------------------------------------------- FIGURE --------------------------------------------
        relative_frequency = 100 * count_array / np.sum(count_array)
        x = np.arange(len(relative_frequency))  # the label locations

        ax.bar(x=x + offset, height=relative_frequency, width=width, label=ds_label, color=colors[counter],
               align='center')

        for j in x:
            ax.text(x=j + offset, y=relative_frequency[j] + width,
                    s=str(count_array[j]), **text_kwargs)
            # s=str(np.round(relative_frequency[j], 1)) + '%', **text_kwargs)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(stations_config.keys(), fontsize=14)
        ax.legend(fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(top=False, right=False)

        plt.xlabel('Lymph node station', fontsize=20)
        plt.ylabel('Frequency (%)', fontsize=20)
        ax.set_ylim(0, 27)
        ax.set_xlim(-0.5, len(relative_frequency) - 0.2)

        # Save figure to reports folder
        fig_path = os.path.join(reports_path, 'figures/')
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(fig_path + 'station_distribution.png', bbox_inches='tight', dpi=300)

        # -------------------------------------------- REPORT --------------------------------------------
        df = []
        for idx, elem in enumerate(count_array):
            df.append([list(stations_config.keys())[idx], elem])
        df.append(['Total', np.sum(count_array)])

        df = pd.DataFrame(df, columns=['Station', 'Files'])
        df.to_csv(reports_path + f'station_distribution_{ds_label}.csv', sep='\t', index=False)
    '''
    if test_ds is not None:
        # Combine the counts from train, validation, and test datasets
        combined_counts = np.zeros(num_stations)

        for ds in (train_ds, val_ds, test_ds):
            count_array = count_station_distribution(ds, num_stations)
            combined_counts += count_array

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(num_stations)  # the label locations
        width = 0.5  # width of bars

        ax.bar(x, height=combined_counts, width=width, color='#80B1D3', align='center')

        # Set text in the middle of the bars
        #for i in range(num_stations):
        #    ax.text(x=x[i], y=combined_counts[i] / 2, s=str(int(combined_counts[i])), ha='center', va='center',
        #            fontsize=12, color='black')

        ax.set_xticks(x)
        ax.set_xticklabels(stations_config.keys(), fontsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(top=False, right=False)

        plt.xlabel('Lymph node station', fontsize=20)
        plt.ylabel('Number of frames', fontsize=20)

        # Handle the case where the maximum value is 0
        if max(combined_counts) > 0:
            ax.set_ylim(0, max(combined_counts) * 1.1)  # Adjust based on maximum value
        else:
            ax.set_ylim(0, 1)  # Set a default limit if all values are zero

        ax.set_xlim(-0.5, num_stations - 0.5)

        # Save figure to reports folder
        fig_path = os.path.join(reports_path, 'figures/')
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(fig_path + 'combined_station_distribution.png', bbox_inches='tight', dpi=1200)
        plt.show()

        # Save report to CSV
        df = pd.DataFrame({
            'Station': list(stations_config.keys()),
            'Files': combined_counts.astype(int)
        })
        df.loc[len(df.index)] = ['Total', np.sum(combined_counts)]

        df.to_csv(os.path.join(reports_path, 'combined_station_distribution.csv'), sep='\t', index=False)

    else:
        # only train and validation
        colors = ['#80B1D3', '#bebada']
        width = 0.4  # width of bars

        for ds_label, batch_gen_i, offset in zip(('train', 'validation'), (train_ds, val_ds), (-width + width/2, width/2)):

            counter = 0 if ds_label == 'train' else 1
            count_array = count_station_distribution(batch_gen_i, num_stations)

            # -------------------------------------------- FIGURE --------------------------------------------
            relative_frequency = 100 * count_array / np.sum(count_array)
            x = np.arange(len(relative_frequency))  # the label locations

            ax.bar(x=x + offset, height=relative_frequency, width=width, label=ds_label, color=colors[counter],
                   align='center')

            for j in x:
                ax.text(x=j + offset, y=relative_frequency[j] + width,
                        s=str(count_array[j]), **text_kwargs)
                # s=str(np.round(relative_frequency[j], 1)) + '%', **text_kwargs)

            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(stations_config.keys(), fontsize=14)
            ax.legend(fontsize=16)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(top=False, right=False)

            plt.xlabel('Lymph node station', fontsize=20)
            plt.ylabel('Frequency (%)', fontsize=20)
            ax.set_ylim(0, 27)
            ax.set_xlim(-0.5, len(relative_frequency) - 0.2)

            # Save figure to reports folder
            fig_path = os.path.join(reports_path, 'figures/')
            os.makedirs(fig_path, exist_ok=True)
            plt.savefig(fig_path + 'station_distribution.png', bbox_inches='tight', dpi=300)

            # -------------------------------------------- REPORT --------------------------------------------
            df = []
            for idx, elem in enumerate(count_array):
                df.append([list(stations_config.keys())[idx], elem])
            df.append(['Total', np.sum(count_array)])

            df = pd.DataFrame(df, columns=['Station', 'Files'])
            df.to_csv(reports_path + f'station_distribution_{ds_label}.csv', sep='\t', index=False)



