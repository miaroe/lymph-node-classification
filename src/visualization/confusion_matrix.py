from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf

plt.style.use('dark_background')

def confusion_matrix_and_report(model, val_ds, num_stations, stations_config, reports_path):
    # Initialize the true labels and predicted labels arrays
    true_labels = []
    pred_labels = []

    for batch in val_ds:
        images, labels = batch #shape=(32, 256, 256, 3), shape=(32,)

        pred_probs = model.predict(images) #shape=(32,)

        #need to convert the probability-based predicted labels into binary format by applying a threshold
        pred = (pred_probs >= 0.5).astype(int).flatten()
        true_labels.extend(labels.numpy())
        pred_labels.extend(pred)

    # -------------------------------------------- FIGURE --------------------------------------------

    #does not include 'other' class
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(stations_config.keys()))
    disp.plot()

    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    disp.figure_.savefig(fig_path + 'confusion_matrix.png')

    # -------------------------------------------- REPORT --------------------------------------------

    report = classification_report(y_true=true_labels,
                                   y_pred=pred_labels,
                                   digits=3,
                                   labels=range(num_stations),
                                   target_names=list(stations_config.keys()),
                                   output_dict=True)
    #save report to csv
    df = pd.DataFrame(report).transpose()
    os.makedirs(reports_path, exist_ok=True)
    df.to_csv(reports_path + 'report.csv', index=True, sep='\t')