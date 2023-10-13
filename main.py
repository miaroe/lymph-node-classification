from tensorflow.python.framework.random_seed import set_seed

from src.resources.config import *
from src.models.training.train_classification import train_model
from src.models.evaluation.evaluate_classification import evaluate_model


def main():
    """ The function running the entire pipeline of the project """
    # To enable determinism between experiments
    #set_seed(42)

    trainer = train_model(data_path=data_path,
                          test_ds_path=test_ds_path,
                          log_path=log_path,
                          image_shape=(img_size, img_size),
                          validation_split=validation_split,
                          test_split=test_split,
                          batch_size=batch_size,
                          stations_config=stations_config,
                          num_stations=num_stations,
                          loss=loss,
                          model_type=model_type,
                          model_arch=model_arch,
                          instance_size=instance_size,
                          learning_rate=learning_rate,
                          model_path=model_path,
                          patience=patience,
                          epochs=epochs,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps,
                          stride=stride,
                          augment=augment,
                          stratified_cv=stratified_cv,
                          seq_length=seq_length)

    evaluate_model(trainer=trainer,
                   reports_path=reports_path,
                   model_path=model_path,
                   visualize_predictions=visualize_predictions,
                   learning_curve=learning_curve,
                   conf_matrix=conf_matrix,
                   model_layout=model_layout,
                   station_distribution=station_distribution,
                   compare_metrics=compare_metrics)


if __name__ == "__main__":
    main()
