import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def station_distribution_figure_and_report(trainer, reports_path):

    batch_generator = trainer.generator.training
    count_array = np.zeros(shape=trainer.pipeline.get_num_stations() , dtype=int)

    for filepath in batch_generator.files:
        dirname = os.path.dirname(filepath)
        sequence_name = os.path.split(dirname)[1]
        station_label = sequence_name.split('_')[1]  # split ['Station', '10R', '001']
        count_array[trainer.pipeline.get_station(station_label)] += 1


    # -------------------------------------------- FIGURE --------------------------------------------
    relative_frequency = 100 * count_array / np.sum(count_array)

    text_kwargs = {'fontsize': 10, 'horizontalalignment': 'center'}
    dy = 0.5

    plt.bar(x=trainer.pipeline.stations_config.keys(), height=relative_frequency)
    for i in range(len(relative_frequency)):
        plt.text(x=i, y=relative_frequency[i] + dy,
                 s=str(np.round(relative_frequency[i], 1)) + '%', **text_kwargs)
    plt.xlabel('Station label', fontsize=16)
    plt.ylabel('Frequency (%)', fontsize=16)
    plt.ylim(0, 20)

    # Save figure to reports folder
    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + trainer.model_arch + '_' + str(trainer.station_config_nr) +  '_station_distribution.png')


    # -------------------------------------------- REPORT --------------------------------------------
    df = []
    for idx, elem in enumerate(count_array):
        df.append([trainer.pipeline.stations_to_label(idx), elem])
    df.append(['Total', np.sum(count_array)])

    df = pd.DataFrame(df, columns=['Station', 'Files'])
    df.to_csv(reports_path + trainer.model_arch + '_' + str(trainer.station_config_nr) + '_station_distribution.csv', sep='\t', index=False)