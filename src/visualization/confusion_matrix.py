from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')

def confusion_matrix_and_report(trainer, batch_generator, reports_path):
    targets_arr = np.zeros(shape=len(batch_generator.files))
    outputs_arr = np.zeros(shape=len(batch_generator.files))

    for step_idx in tqdm(range(batch_generator.steps_per_epoch), 'Batches'):
        inputs, targets = next(batch_generator)

        outputs = trainer.model.predict(inputs, batch_size=batch_generator.batch_size)
        # categorical = to_categorical(np.argmax(outputs, axis=-1), pipeline.num_classes)

        idx = step_idx * batch_generator.batch_size
        targets_arr[idx:idx + batch_generator.batch_size] = np.argmax(targets, axis=-1)
        outputs_arr[idx:idx + batch_generator.batch_size] = np.argmax(outputs, axis=-1)

    cm = confusion_matrix(targets_arr, outputs_arr, labels=range(trainer.pipeline.get_num_stations()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=trainer.pipeline.stations_config.keys())
    disp.plot()
    fig_name = trainer.model_arch + '_confusion_matrix.png'
    disp.figure_.savefig(os.path.join(reports_path, 'figures/') + fig_name)

    report = classification_report(y_true=targets_arr,
                                   y_pred=outputs_arr,
                                   digits=3,
                                   labels=range(trainer.pipeline.get_num_stations()),
                                   target_names=trainer.pipeline.stations_config.keys(),
                                   output_dict=True)
    #save report to csv
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(reports_path, trainer.model_arch + '_report.csv'), index=True, sep='\t')