import os
from tensorflow.keras.metrics import Precision, Recall

from src.resources.loss import get_loss
from src.resources.ml_models import get_arch
from src.utils.get_current_dir import get_latest_date_time
from src.visualization.learning_curve import plot_learning_curve
from src.visualization.prediction_model import plot_predictions
from src.visualization.visual_model import visual_model
from src.visualization.confusion_matrix import confusion_matrix_and_report
from src.visualization.station_distribution import station_distribution_figure_and_report
from src.visualization.compare_metrics import plot_compare_metrics
from src.resources.train_config import get_config


# -----------------------------  EVALUATING ----------------------------------
def evaluate_model(trainer, reports_path, model_path, visualize_predictions, learning_curve, conf_matrix, model_layout,
                   station_distribution, compare_metrics):
    print("Evaluating model: " + model_path)

    config_path = os.path.join(model_path, 'config.json')

    config = get_config(config_path)
    train_config = config["train_config"]


    model = get_arch(train_config.get('model_arch'), train_config.get('instance_size'),
                     train_config.get('num_stations'), train_config.get('seq_length'))

    model.compile(loss=get_loss(train_config.get('loss')), optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
                  

    model.load_weights(filepath=os.path.join(model_path, 'best_model')).expect_partial()

    score = trainer.model.evaluate(trainer.val_ds, return_dict=True)

    print("Model metrics using validation dataset: ")
    print(f'{"Metric":<12}{"Value"}')
    for metric, value in score.items():
        print(f'{metric:<12}{value:<.4f}')

    if visualize_predictions:
        plot_predictions(model, trainer.val_ds, trainer.pipeline.station_names, reports_path)

    # save learning curve to src/reports/figures
    if learning_curve:
        plot_learning_curve(model_path, reports_path)

    if station_distribution:
        station_distribution_figure_and_report(trainer.train_ds, trainer.val_ds, train_config.get('num_stations'),
                                               train_config.get('stations_config'), reports_path)

    # save confusion matrix to src/reports/figures and save classification report to src/reports
    if conf_matrix:
        confusion_matrix_and_report(model, trainer.val_ds, train_config.get('num_stations'), #TODO: chnage back to val_ds
                                    train_config.get('stations_config'),
                                    reports_path)

    if compare_metrics:
        current_dir = get_latest_date_time('/home/miaroe/workspace/lymph-node-classification/output/models/')
        print('current_dir', current_dir)
        model_paths = ['/home/miaroe/workspace/lymph-node-classification/output/models/' + model for model in
                       [current_dir]]  # to compare multiple models, add more paths manually
        model_names = [trainer.model_arch]  # saved in filename
        plot_compare_metrics(model_paths, model_names, reports_path)

    # save model layout to src/reports/figures
    if model_layout:
        visual_model(model, reports_path)
