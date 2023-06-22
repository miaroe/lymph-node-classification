from src.resources.config import *
from mlmia import TrainConfig
from src.utils.json_parser import parse_json

os.makedirs(model_path, exist_ok=True)

config = TrainConfig(batch_size=batch_size,
                     model_arch=model_arch,
                     instance_size=instance_size,
                     loss=loss,
                     station_config_nr=station_config_nr,
                     data_path=data_path,
                     image_shape=(img_size, img_size),
                     early_stop_patience=patience,
                     log_directory=model_path,
                     num_epochs=epochs,
                     validation_split=validation_split,
                     test_split=test_split)

def set_train_config():
    return config

def get_train_config(config_path):
    json_config = parse_json(config_path)
    return json_config["train_config"]