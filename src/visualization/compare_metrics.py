import matplotlib.pyplot as plt
import os
import re
from src.utils.json_parser import parse_json

plt.style.use('dark_background')

def plot_compare_metrics(model_paths, reports_path):

    # create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Training and Validation Metrics')

    # iterate over the history files
    for model_path in (model_paths):
        results = parse_json(os.path.join(model_path + "/results.json"))['0']

        train_loss = [entry['loss'] for entry in results]
        val_loss = [entry['val_loss'] for entry in results]
        train_accuracy = [entry['accuracy'] for entry in results]
        val_accuracy = [entry['val_accuracy'] for entry in results]
        train_precision = [entry['precision'] for entry in results]
        val_precision = [entry['val_precision'] for entry in results]
        train_recall = [entry['recall'] for entry in results]
        val_recall = [entry['val_recall'] for entry in results]
        epochs = range(1, len(train_loss) + 1)

        model_name = os.path.basename(model_path)

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
