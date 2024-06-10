import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from matplotlib.legend_handler import HandlerTuple

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from src.data.db.db_clinician_test import get_clinician_test_df

def get_metrics_dfs():

    stations = ['4L', '4R', '7L', '7R', '10L', '10R', '11L', '11R']

    # create df with station labels as columns
    precision_df = pd.DataFrame(columns=['4L', '4R', '7L', '7R', '10L', '10R', '11L', '11R', 'Total'])
    recall_df = pd.DataFrame(columns=['4L', '4R', '7L', '7R', '10L', '10R', '11L', '11R', 'Total'])

    # get clinician test df
    clinician_df = get_clinician_test_df()

    # get y_true and y_pred for all and calulate confusion matrix using stations to get the order of the labels
    y_true = clinician_df['true_label'].tolist()
    y_pred = clinician_df['predicted_label'].tolist()
    confusion_mat = confusion_matrix(y_true, y_pred, labels=stations)
    print(confusion_mat)
    # plot confusion matrix using confusionmatrixdisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=stations)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:}')




    for username in clinician_df['username'].unique():
        print(f'Username: {username}')
        y_true = clinician_df[clinician_df['username'] == username]['true_label'].tolist()
        y_pred = clinician_df[clinician_df['username'] == username]['predicted_label'].tolist()

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        #print(f'Accuracy: {accuracy:}')

        # Calculate precision
        precision = precision_score(y_true, y_pred, average='weighted')
        #print(f'Precision: {precision:}')

        # Calculate recall
        recall = recall_score(y_true, y_pred, average='weighted')
        #print(f'Recall: {recall:}')

        # Calculate classification report
        classification_rep_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()

        # remove the last three rows (avg)
        classification_rep_df = classification_rep_df.iloc[:-3, :]

        # add total row
        classification_rep_df.loc['Total', 'precision'] = precision
        classification_rep_df.loc['Total', 'recall'] = recall

        # append to precision and recall df and name each row by username
        precision_df = precision_df.append(classification_rep_df['precision'].rename('Clinician'))
        recall_df = recall_df.append(classification_rep_df['recall'].rename('Clinician'))

    return precision_df, recall_df


def create_box_plot():

    # Get the precision and recall DataFrames
    precision_df, recall_df = get_metrics_dfs()
    print(recall_df)
    # Reshape the data for seaborn boxplot
    precision_long = precision_df.melt(var_name='Station', value_name='Clinician precision', ignore_index=False)
    recall_long = recall_df.melt(var_name='Station', value_name='Clinician recall', ignore_index=False)

    # Combine precision and recall into a single DataFrame
    combined_df = pd.concat([precision_long, recall_long['Clinician recall']], axis=1)

    # Melt the combined DataFrame for easier plotting
    combined_long = combined_df.melt(id_vars='Station', var_name='Metric', value_name='Score')

    precision_values_CNN = [1.0, 0.5, 0.667, 0.444, 0.0, 1.0, 1.0, 0.5, 0.724]
    recall_values_CNN = [0.667, 0.667, 0.5, 1.0, 0.0, 0.25, 0.25, 0.6, 0.581]

    values_df = pd.DataFrame({'Station': precision_df.columns, 'CNN precision': precision_values_CNN, 'CNN recall': recall_values_CNN})

    #melt the values_df for easier plotting
    values_long = values_df.melt(id_vars='Station', var_name='Metric', value_name='Score')

    plt.figure(figsize=(18, 8))
    sns.set(style="whitegrid")
    ax = sns.boxplot(x='Station', y='Score', hue='Metric', data=combined_long,
                     palette={'Clinician precision': '#8DD3C7', 'Clinician recall': '#AFA9DA'}, boxprops={'alpha': 0.7}, showfliers=False)

    # Add the data points used to create the boxplot as scatter points
    sns.stripplot(x='Station', y='Score', hue='Metric', data=combined_long, dodge=True, jitter=True, size=7,
                  linewidth=0.8, palette={'Clinician precision': '#8DD3C7', 'Clinician recall': '#AFA9DA'}, marker='o',
                  ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles=[(handles[0], handles[2]), (handles[1], handles[3])],
              labels=['Clinician precision', 'Clinician recall'],
              loc='upper right', handlelength=4, fontsize=18,
              handler_map={tuple: HandlerTuple(ndivide=None)})

    for lh in leg.legendHandles:
        lh.set_edgecolor('k')

    # Add the CNN values from the values_long to the plot as points, centered on the boxes
    #sns.stripplot(x='Station', y='Score', hue='Metric', data=values_long, dodge=True, jitter=False, size=10, linewidth=2, palette={'CNN precision': '#fb8072', 'CNN recall': '#bebada'})

    # Add labels
    plt.xlabel('Station', fontsize=20)
    plt.ylabel('Score', fontsize=20)

    # Increase the fontsizes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # change y-axis to percentage
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))


    fig_path = '/home/miaroe/workspace/lymph-node-classification/figures/'
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + 'box_plot.png', bbox_inches='tight')


create_box_plot()

