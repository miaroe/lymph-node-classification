from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')


def confusion_matrix_and_report(pipeline, model, batch_generator, reports_path):
    targets_arr = np.zeros(shape=len(batch_generator.files), dtype=int)
    outputs_arr = np.zeros(shape=len(batch_generator.files), dtype=int)

    for step_idx in tqdm(range(batch_generator.steps_per_epoch), 'Batches'):
        inputs, targets = next(batch_generator)

        outputs = model.predict(inputs, batch_size=inputs.shape[0])

        idx = step_idx * batch_generator.batch_size
        targets_arr[idx:idx + inputs.shape[0]] = np.argmax(targets, axis=-1)
        outputs_arr[idx:idx + inputs.shape[0]] = np.argmax(outputs, axis=-1)

    # -------------------------------------------- FIGURE --------------------------------------------

    cm = confusion_matrix(targets_arr, outputs_arr, labels=range(pipeline.get_num_stations()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=pipeline.stations_config.keys())
    disp.plot()

    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    disp.figure_.savefig(fig_path + 'confusion_matrix.png')

    # -------------------------------------------- REPORT --------------------------------------------

    report = classification_report(y_true=targets_arr,
                                   y_pred=outputs_arr,
                                   digits=3,
                                   labels=range(pipeline.get_num_stations()),
                                   target_names=pipeline.stations_config.keys(),
                                   output_dict=True)
    #save report to csv
    df = pd.DataFrame(report).transpose()
    os.makedirs(reports_path, exist_ok=True)
    df.to_csv(reports_path + 'report.csv', index=True, sep='\t')