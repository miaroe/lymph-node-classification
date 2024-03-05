import numpy as np
import tensorflow as tf

from src.models.segmentation_masks.create_masks import create_seg_masks
from src.resources.config import *
from src.models.training.train_classification import train_model
from src.models.evaluation.evaluate_classification import evaluate_model

def main():
    """ The function running the entire pipeline of the project """
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    if model_type == 'sequence_with_segmentation' and perform_segmentation:
        create_seg_masks(old_data_path=old_data_path,
                         data_path=data_path,
                         seg_model_path=seg_model_path
                         )


    trainer = train_model(data_path=data_path,
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
                          set_stride=set_stride,
                          augment=augment,
                          stratified_cv=stratified_cv,
                          seq_length=seq_length,
                          full_video=full_video,
                          use_quality_weights=use_quality_weights,
                          use_gen=use_gen)

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
