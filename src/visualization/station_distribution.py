import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.count_station_distribution import count_station_distribution


def station_distribution_figure_and_report(pipeline, batch_generator, reports_path):
    fig, ax = plt.subplots(figsize=(16, 10))
    text_kwargs = {'fontsize': 13, 'horizontalalignment': 'center'}
    width = 0.4  # width of bars

    for label, batch_gen_i, offset in zip(('train', 'val'), (batch_generator.training, batch_generator.validation), (0, width)):

        count_array = count_station_distribution(pipeline, batch_gen_i)

        # -------------------------------------------- FIGURE --------------------------------------------
        relative_frequency = 100 * count_array / np.sum(count_array)
        x = np.arange(len(relative_frequency))  # the label locations

        ax.bar(x=x + offset, height=relative_frequency, width=width, label=label)

        for j in x:
            ax.text(x=j + offset, y=relative_frequency[j] + width,
                     s=str(np.round(relative_frequency[j], 1)) + '%', **text_kwargs)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(pipeline.stations_config.keys(), fontsize=14)
        ax.legend(fontsize=16)

        plt.xlabel('Station label', fontsize=20)
        plt.ylabel('Frequency (%)', fontsize=20)
        plt.ylim(0, 25)

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
        df.to_csv(reports_path + f'station_distribution_{label}.csv', sep='\t', index=False)

