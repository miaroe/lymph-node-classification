import matplotlib.pyplot as plt
import os
import pandas as pd
import re

plt.style.use('dark_background')

def plot_compare_metrics(history_path, model_names, reports_path):

    # create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Training and Validation Metrics')

    # iterate over the history files
    for model_name in (model_names):
        history = pd.read_csv(os.path.join(history_path, model_name + "-training_history.csv"))
        train_loss = history['loss'].to_numpy()
        val_loss = history['val_loss'].to_numpy()
        train_accuracy = history['accuracy'].to_numpy()
        val_accuracy = history['val_accuracy'].to_numpy()
        train_precision = history['precision'].to_numpy()
        val_precision = history['val_precision'].to_numpy()
        train_recall = history['recall'].to_numpy()
        val_recall = history['val_recall'].to_numpy()
        epochs = range(1, len(train_loss) + 1)

        model_name = re.search(r"Loss-(.*?)_", model_name).group(1)

        # plot and label the training and validation loss values
        axs[0, 0].plot(epochs, train_loss, label=model_name + ' Training Loss')
        axs[0, 0].plot(epochs, val_loss, label=model_name + ' Validation Loss')
        axs[0, 1].plot(epochs, train_accuracy, label=model_name + ' Training Accuracy')
        axs[0, 1].plot(epochs, val_accuracy, label=model_name + ' Validation Accuracy')
        axs[1, 0].plot(epochs, train_precision, label=model_name + ' Training Precision')
        axs[1, 0].plot(epochs, val_precision, label=model_name + ' Validation Precision')
        axs[1, 1].plot(epochs, train_recall, label=model_name + ' Training Recall')
        axs[1, 1].plot(epochs, val_recall, label=model_name + ' Validation Recall')

    # add in a title and axes labels
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Accuracy')
    axs[1, 0].set_title('Precision')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 1].set_title('Recall')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Recall')

    # add a legend
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()

    # save the figure
    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + 'compare_metrics.png')
