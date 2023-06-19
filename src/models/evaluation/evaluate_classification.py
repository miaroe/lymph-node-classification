from src.visualization.learning_curve import plot_learning_curve
from src.visualization.visual_model import visual_model
from src.visualization.confusion_matrix import confusion_matrix_and_report
from src.visualization.station_distribution import station_distribution_figure_and_report
from src.visualization.compare_metrics import plot_compare_metrics

# -----------------------------  EVALUATING ----------------------------------
def evaluate_model(trainer, reports_path, learning_curve, conf_matrix, model_layout, station_distribution, compare_metrics):
    print("Evaluating model: " + trainer.model_path + trainer.model_name)

    trainer.model.load_weights(trainer.model_path + trainer.model_name)
    batch_generator = trainer.generator.validation
    '''
    score = trainer.model.evaluate(batch_generator,
                               steps=batch_generator.steps_per_epoch,
                               return_dict=True
                               )
    print(f'{"Metric":<12}{"Value"}')
    for metric, value in score.items():
        print(f'{metric:<12}{value:<.4f}')
    
    
    x_test, y_test = trainer.generator.testing.load_dataset()
    pred_test = trainer.model.evaluate(x_test, y_test, return_dict=True)
    print({"test_{}".format(key): val for key, val in pred_test.items()})
    print(pred_test)
'''
    # save learning curve to src/reports/figures
    if learning_curve:
        plot_learning_curve(trainer.history_path, trainer.model_name, reports_path)

    # save confusion matrix to src/reports/figures and save classification report to src/reports
    if conf_matrix:
        confusion_matrix_and_report(trainer, batch_generator, reports_path)

    # save model layout to src/reports/figures
    if model_layout:
        visual_model(trainer.model, reports_path, trainer.model_arch)

    if station_distribution:
        station_distribution_figure_and_report(trainer, reports_path)

    if compare_metrics:
        model_names = ['Arch-vgg16_2023-06-19_11:48:01_Stations_config_nr-1_Epochs-50_Loss-categoricalCrossentropy_ImageSize-256_BatchSize-32_Augmentation-True_ValPercent-20',
                       'Arch-vgg16_Stations_config_nr-4_2023-06-16_15:59:03_Epochs-100_Loss-focalLoss_ImageSize-256_BatchSize-32_Augmentation-True_ValPercent-20']
        plot_compare_metrics(trainer.history_path, model_names, reports_path)
