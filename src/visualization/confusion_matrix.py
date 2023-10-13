from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')


def confusion_matrix_and_report(true_labels, predictions, num_stations, stations_config, reports_path):
    # -------------------------------------------- FIGURE --------------------------------------------

    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(stations_config.keys()))
    disp.plot()

    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    disp.figure_.savefig(fig_path + 'confusion_matrix.png')

    # -------------------------------------------- REPORT --------------------------------------------

    report = classification_report(y_true=true_labels,
                                   y_pred=predictions,
                                   digits=3,
                                   labels=range(num_stations),
                                   target_names=list(stations_config.keys()),
                                   output_dict=True)
    # save report to csv
    df = pd.DataFrame(report).transpose()
    os.makedirs(reports_path, exist_ok=True)
    df.to_csv(reports_path + 'report.csv', index=True, sep='\t')
