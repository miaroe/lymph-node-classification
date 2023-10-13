import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report


class ClassMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, station_names, train_ds, val_ds, save_path):
        super(ClassMetricsCallback, self).__init__()
        self.save_path = save_path
        self.station_names = station_names
        self.num_stations = len(station_names)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.metrics_file = open(self.save_path, 'w')
        self.metrics_file.write('Epoch,Class,Precision,Recall,F1-Score\n')
        self.pred_file = open('val_preds.txt', 'w')


    def on_epoch_end(self, epoch, logs=None):
        train_labels = []
        val_preds = []
        val_labels = []

        for train_batch in self.train_ds:
            train_images, train_true_labels = train_batch
            train_labels.extend(train_true_labels)

        for val_batch in self.val_ds:
            val_images, val_true_labels = val_batch
            val_preds_batch = self.model.predict(val_images)
            val_preds.extend(val_preds_batch)
            val_labels.extend(val_true_labels)

        #print('val_preds_num: ', val_preds)
        # save val_preds to file
        for pred in val_preds:
            self.pred_file.write(f'{pred}\n')

        train_labels = np.argmax(np.array(train_labels), axis=1)
        val_preds = np.argmax(np.array(val_preds), axis=1)
        val_labels = np.argmax(np.array(val_labels), axis=1)
        #print('val_preds: ', val_preds)
        #print('train_labels: ', train_labels)
        #print('val_labels: ', val_labels)

        class_metrics = classification_report(y_true=val_labels, y_pred=val_preds, digits=3,
                                              labels=range(self.num_stations),
                                              target_names=self.station_names, output_dict=True)

        for class_name, metrics in class_metrics.items():
            if class_name != 'accuracy' and class_name != 'macro avg' and class_name != 'weighted avg' and class_name != 'micro avg':
                precision = metrics['precision']
                recall = metrics['recall']
                self.metrics_file.write(f'{epoch},{class_name},{precision},{recall}\n')



    def on_train_end(self, logs=None):
        self.metrics_file.close()
        self.pred_file.close()
