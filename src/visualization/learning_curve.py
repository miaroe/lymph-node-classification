import matplotlib.pyplot as plt
import os
import pandas as pd

plt.style.use('dark_background')

def plot_learning_curve(history_path, model_name, reports_path):
    history = pd.read_csv(os.path.join(history_path, model_name + "-training_history.csv"))
    train_loss = history['loss'].to_numpy()
    val_loss = history['val_loss'].to_numpy()
    epochs = range(1, len(train_loss) + 1)

    fig = plt.figure(figsize=(10, 5))

    # Plot and label the training and validation loss values
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    #plt.xticks(arange(0, 21, 2))

    plt.legend(loc='best')

    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + model_name + '_learning_curve.png')
