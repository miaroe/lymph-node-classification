from tensorflow.python.framework.random_seed import set_seed

from src.resources.config import *
from src.models.training.train_classification import train_model
from src.models.evaluation.evaluate_classification import evaluate_model
from src.visualization.station_distribution import station_distribution_figure_and_report


def main():
    """ The function running the entire pipeline of the project """
    # To enable determinism between experiments
    set_seed(42)

    if perform_training:
        train_model(stratified_cv=stratified_cv,
                    data_path=data_path,
                    log_path = log_path,
                    image_shape=(img_size, img_size),
                    tf_dataset=tf_dataset,
                    validation_split=validation_split,
                    test_split=test_split,
                    batch_size=batch_size,
                    split_by=split_by,
                    station_config_nr=station_config_nr,
                    loss = loss,
                    augment_data=augment_data,
                    model_arch=model_arch,
                    instance_size=instance_size,
                    learning_rate=learning_rate,
                    model_path=model_path,
                    history_path=history_path,
                    model_name=model_name,
                    patience=patience,
                    epochs=epochs)

    evaluate_model(reports_path=reports_path,
                   model_path=model_path,
                   learning_curve=learning_curve,
                   conf_matrix=conf_matrix,
                   model_layout=model_layout,
                   station_distribution=station_distribution,
                   compare_metrics=compare_metrics)



if __name__ == "__main__":
    main()

