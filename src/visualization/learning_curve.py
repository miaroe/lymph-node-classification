import matplotlib.pyplot as plt
import os
from src.utils.json_parser import parse_json
import pandas as pd

def plot_learning_curve(model_path, reports_path):
    """
    Plot the learning curve for a given model
    :param model_path:
    :param reports_path:
    :return:
    """
    plt.style.use('classic')
    results = parse_json(os.path.join(model_path + "results.json"))['0']

    train_loss = [entry['loss'] for entry in results]
    val_loss = [entry['val_loss'] for entry in results]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#8dd3c7', '#fdb462']

    # Plot and label the training and validation loss values
    plt.plot(epochs, train_loss, label='Training Loss', color=colors[0], linewidth=3)
    plt.plot(epochs, val_loss, label='Validation Loss', color=colors[1], linewidth=3)

    # Add axes labels
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)

    # Set the ticks with an interval of 5
    ax.set_xticks(range(0, len(train_loss), 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False)

    plt.legend(loc='best')

    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + 'learning_curve2.png', dpi=300, bbox_inches='tight')

plot_learning_curve('/home/miaroe/workspace/lymph-node-classification/output/models/2023-11-22/10:38:04/', '/home/miaroe/workspace/lymph-node-classification/reports/2023-11-22/10:38:04/')