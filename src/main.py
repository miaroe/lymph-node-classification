from tensorflow.python.framework.random_seed import set_seed

from src.resources.config import *
from models.training.train_classification import train_model
from models.evaluation.evaluate_classification import evaluate_model

def main():
    """ The function running the entire pipeline of the project """
    # To enable determinism between experiments
    set_seed(42)

    trainer = train_model(perform_training=perform_training,
                          data_path=data_path,
                          image_shape=(img_size, img_size),
                          tf_dataset=tf_dataset,
                          validation_split=validation_split,
                          batch_size=batch_size,
                          split_by=split_by,
                          samples_per_load=samples_per_load,
                          station_config_nr=station_config_nr,
                          augment_data=augment_data,
                          model_arch =model_arch,
                          instance_size=instance_size,
                          learning_rate=learning_rate,
                          model_path=model_path,
                          history_path=history_path,
                          model_name=model_name,
                          patience=patience,
                          epochs=epochs)

    evaluate_model(trainer=trainer,
                   reports_path=reports_path,
                   conf_matrix=conf_matrix,
                   model_layout=model_layout)

if __name__ == "__main__":
    main()

