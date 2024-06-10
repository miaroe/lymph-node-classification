import numpy as np
import tensorflow as tf

from src.resources.config import *
from src.models.training.train_classification import train_model
from src.models.evaluation.evaluate_classification import evaluate_model

def main_cv():
    """ The function running the entire pipeline of the project """
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    n_splits = 5
    for fold in range(n_splits):

        trainer = train_model(data_path=os.path.join(data_path, f'fold_{fold}_v2'),
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
                              model_path=os.path.join(model_path, f'fold_{fold}_v2' + '/'),
                              patience=patience,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              set_stride=set_stride,
                              augment=augment,
                              stratified_cv=stratified_cv,
                              seq_length=seq_length,
                              full_video=full_video,
                              use_quality_weights=use_quality_weights,
                              use_gen=use_gen)

        evaluate_model(trainer=trainer,
                       reports_path=os.path.join(reports_path, f'fold_{fold}_v2' + '/'),
                       model_path=os.path.join(model_path, f'fold_{fold}_v2' + '/'),
                       visualize_predictions=visualize_predictions,
                       learning_curve=learning_curve,
                       conf_matrix=conf_matrix,
                       model_layout=model_layout,
                       station_distribution=station_distribution,
                       compare_metrics=compare_metrics)

        # clear memory
        tf.keras.backend.clear_session()
        break



if __name__ == "__main__":
    main_cv()
