from tensorflow.python.framework.random_seed import set_seed

from src.resources.config import *
from models.training.train_classification import train_model
from models.evaluation.evaluate_classification import evaluate_model
from src.utils import logger

logger.init_logger('train_classification')
def main():
    """ The function running the entire pipeline of the project """
    # To enable determinism between experiments
    set_seed(42)

    trainer = train_model(perform_training=perform_training,
                          data_path=data_path,
                          image_shape=(img_size, img_size),
                          tf_dataset=True,
                          validation_split=validation_split,
                          batch_size=batch_size,
                          split_by='stations',
                          samples_per_load=10,
                          label_config=label_config,
                          augment_data=augment_data,
                          model_arch =model_arch,
                          instance_size=instance_size,
                          learning_rate=learning_rate,
                          save_model_path=save_model_path,
                          history_path=history_path,
                          model_name=model_name,
                          patience=patience,
                          epochs=epochs)

    evaluate_model(trainer=trainer)

if __name__ == "__main__":
    main()

logger.close_logger()