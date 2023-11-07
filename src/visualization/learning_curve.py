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
    plt.style.use('ggplot')
    results = parse_json(os.path.join(model_path + "results.json"))['0']

    train_loss = [entry['loss'] for entry in results]
    val_loss = [entry['val_loss'] for entry in results]
    epochs = range(1, len(train_loss) + 1)

    fig = plt.figure(figsize=(10, 5))
    colors = ['#fb8072', '#fdb462']

    # Plot and label the training and validation loss values
    plt.plot(epochs, train_loss, label='Training Loss', color=colors[0])
    plt.plot(epochs, val_loss, label='Validation Loss', color=colors[1])

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    #plt.xticks(arange(0, 21, 2))

    plt.legend(loc='best')

    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + 'learning_curve.png', dpi=300, bbox_inches='tight')
