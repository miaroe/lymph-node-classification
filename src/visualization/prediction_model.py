import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np


def get_predictions(model, test_ds):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_ds)
    return predictions


def plot_predictions(model, test_ds, model_type, station_names, reports_path):
    print('All predictions: ')
    print(get_predictions(model, test_ds))
    print('station names: ', station_names)

    # plot the first 9 images in the test set, and their predicted labels and true labels
    fig = plt.figure(figsize=(10, 10))
    for images, labels in test_ds.take(1):
        predictions = get_predictions(model, images)
        for i in range(len(predictions)):
            print('predictions[i]: ', predictions[i])
            print('np.argmax(labels[i]): ', np.argmax(labels[i]))
            ax = plt.subplot(3, 3, i + 1)
            if model_type == 'baseline':
                image = images[i]
            else:  # model_type == 'sequence' and has an extra dimension
                image = images[0][i]
            plt.imshow(image, norm=plt.Normalize(vmin=0, vmax=255))
            if len(station_names) > 2:
                plt.title("Predicted: {}, True: {}".format(station_names[np.argmax(predictions[i])],
                                                           station_names[np.argmax(labels[i])]))
            else:
                plt.title("Predicted: {}, True: {}".format(station_names[int(predictions[i][0])],
                                                           station_names[labels.numpy()[i]]))
            plt.axis("off")

    # save figure
    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + 'predictions.png')
