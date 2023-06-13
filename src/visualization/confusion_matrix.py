from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

def confusion_matrix_and_report(trainer, batch_generator, batch_size, reports_path):
    targets_arr = np.zeros(shape=len(batch_generator.files))
    outputs_arr = np.zeros(shape=len(batch_generator.files))

    for step_idx in tqdm(range(batch_generator.steps_per_epoch), 'Batches'):
        inputs, targets = next(batch_generator)

        outputs = trainer.model.predict(inputs, batch_size=batch_size)
        # categorical = to_categorical(np.argmax(outputs, axis=-1), pipeline.num_classes)

        idx = step_idx * batch_size
        targets_arr[idx:idx + batch_size] = np.argmax(targets, axis=-1)
        outputs_arr[idx:idx + batch_size] = np.argmax(outputs, axis=-1)

    cm = confusion_matrix(targets_arr, outputs_arr, labels=range(trainer.pipeline.get_num_stations()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=trainer.pipeline.stations_config.keys())
    disp.plot()
    plt.show()

    report = classification_report(y_true=targets_arr,
                                   y_pred=outputs_arr,
                                   digits=3,
                                   labels=range(trainer.pipeline.get_num_stations()),
                                   target_names=trainer.pipeline.stations_config.keys(),
                                   output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(reports_path + trainer.model_arch + '_report.csv', index=False)