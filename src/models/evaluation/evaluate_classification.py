import os
from tensorflow.keras.metrics import Precision, Recall

from src.resources.loss_dict import get_loss
from src.visualization.learning_curve import plot_learning_curve
from src.visualization.visual_model import visual_model
from src.visualization.confusion_matrix import confusion_matrix_and_report
from src.visualization.station_distribution import station_distribution_figure_and_report
from src.visualization.compare_metrics import plot_compare_metrics
from src.data.classification_pipeline import EBUSClassificationPipeline
from src.resources.ml_models import get_arch
from src.resources.train_config import get_train_config

# -----------------------------  EVALUATING ----------------------------------
def evaluate_model(reports_path, model_path, learning_curve, conf_matrix, model_layout,
                   station_distribution, compare_metrics):
    print("Evaluating model: " + model_path)

    config_path = os.path.join(model_path, 'config.json')


    pipeline = EBUSClassificationPipeline.from_config(config_path)
    train_config = get_train_config(config_path)

    model = get_arch(train_config.get('model_arch'), train_config.get('instance_size'), pipeline.get_num_stations())
    model.compile(loss=get_loss(train_config.get('loss')), optimizer='adam', metrics=['accuracy', Precision(), Recall()])
    model.load_weights(filepath=os.path.join(model_path, 'best_model')).expect_partial()

    batch_generator = pipeline.generator_containers[0]
    '''
    score = model.evaluate(batch_generator.validation,
                               steps=batch_generator.validation.steps_per_epoch,
                               return_dict=True
                               )
    print(f'{"Metric":<12}{"Value"}')
    for metric, value in score.items():
        print(f'{metric:<12}{value:<.4f}')

    
    x_test, y_test = batch_generator.get_test_data()
    pred_test = model.evaluate(x_test, y_test, return_dict=True)
    print({"test_{}".format(key): val for key, val in pred_test.items()})
    print(pred_test)
    '''
    # save learning curve to src/reports/figures
    if learning_curve:
        plot_learning_curve(model_path, reports_path)

    # save confusion matrix to src/reports/figures and save classification report to src/reports
    if conf_matrix:
        confusion_matrix_and_report(pipeline, model, batch_generator.validation, reports_path)

    # save model layout to src/reports/figures
    if model_layout:
        visual_model(model, reports_path)

    if station_distribution:
        station_distribution_figure_and_report(pipeline, batch_generator, reports_path)

    if compare_metrics:
        model_paths = ['/home/miaroe/workspace/lymph-node-classification/output/models/' + model for model in ['2023-06-23/16:51:46', '2023-06-23/16:51:46']]
        plot_compare_metrics(model_paths, reports_path)

