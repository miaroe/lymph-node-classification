from src.resources.config import *
from mlmia import TrainConfig
from src.utils.json_parser import parse_json

config = TrainConfig(batch_size=batch_size,
                     model_arch=model_arch,
                     instance_size=instance_size,
                     loss=loss,
                     seq_length=seq_length,
                     set_stride=set_stride,
                     full_video=full_video,
                     steps_per_epoch=steps_per_epoch,
                     validation_steps=validation_steps,
                     augment=augment,
                     filter_data=filter_data,
                     stations_config=stations_config,
                     num_stations=num_stations,
                     data_path=data_path,
                     image_shape=(img_size, img_size),
                     early_stop_patience=patience,
                     log_directory=log_path,
                     model_directory=model_path,
                     num_epochs=epochs,
                     validation_split=validation_split,
                     test_split=test_split,
                     use_gen=use_gen,
                     perform_segmentation=perform_segmentation,
                     num_train=num_train,
                     num_val=num_val,
                     use_quality_weights=use_quality_weights,
                     learning_rate=learning_rate)



def set_train_config():
    return config


def get_config(config_path):
    json_config = parse_json(config_path)
    return json_config
