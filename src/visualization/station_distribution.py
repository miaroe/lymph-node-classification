import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.count_station_distribution import count_station_distribution


def station_distribution_figure_and_report(pipeline, batch_generator, reports_path):

    for label, batch_gen_i in zip(('train', 'val'), (batch_generator.training, batch_generator.validation)):

        count_array = count_station_distribution(pipeline, batch_gen_i)

        # -------------------------------------------- FIGURE --------------------------------------------
        relative_frequency = 100 * count_array / np.sum(count_array)

        text_kwargs = {'fontsize': 10, 'horizontalalignment': 'center'}
        dy = 0.5

        plt.figure(figsize=(10, 6))
        plt.bar(x=pipeline.stations_config.keys(), height=relative_frequency)
        for i in range(len(relative_frequency)):
            plt.text(x=i, y=relative_frequency[i] + dy,
                     s=str(np.round(relative_frequency[i], 1)) + '%', **text_kwargs)
        plt.xlabel('Station label', fontsize=16)
        plt.ylabel('Frequency (%)', fontsize=16)
        plt.ylim(0, 20)

        # Save figure to reports folder
        fig_path = os.path.join(reports_path, 'figures/')
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(fig_path + 'station_distribution.png')


        # -------------------------------------------- REPORT --------------------------------------------
        df = []
        for idx, elem in enumerate(count_array):
            df.append([pipeline.stations_to_label(idx), elem])
        df.append(['Total', np.sum(count_array)])

        df = pd.DataFrame(df, columns=['Station', 'Files'])
        df.to_csv(reports_path + 'station_distribution.csv', sep='\t', index=False)