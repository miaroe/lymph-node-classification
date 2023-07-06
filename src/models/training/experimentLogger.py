import json
import os
import tensorflow as tf
import numpy as np

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from tensorflow.keras.callbacks import Callback
from collections import UserString

from src.utils.logging_utilities import create_markdown_from_dict

class ExperimentLogger(Callback):
    """ExperimentLogger. Used to log the results of an experiment to a file and to mlflow/tensorboard.

    Parameters
    ----------
        logdir : str
            Path to the directory where the results should be saved.
        pipeline_config : dict
            Dictionary containing the configuration of the pipeline.
        train_config : dict
            Dictionary containing the configuration of the training.
        mlflow : bool
            If true, the results will be logged to mlflow.
        tensorboard : bool
            If true, the results will be logged to tensorboard.
    """
    def __init__(self, logdir: str = None, pipeline_config: dict = None, train_config: dict = None,
                 mlflow: bool = False, tensorboard: bool = False, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if logdir is None:
            self.logdir = Path('logs_' + datetime.now().strftime('%Y-%m-%d-%H-%M'))
        else:
            self.logdir = Path(logdir)

        if not self.logdir.exists():
            self.logdir.mkdir(parents=True)

        self.save_list = []
        self.save_path = os.path.join(self.logdir, 'config.json')
        self.config = {}
        self.result_log = {}
        self.current_fold_name = 0

        self.mlflow_enabled = mlflow
        self.tensorboard_enabled = tensorboard

        if train_config is not None:
            self.config["train_config"] = train_config

        if pipeline_config is not None:
            for key, val in pipeline_config.items():
                self.config[key] = val

        if self.mlflow_enabled:
            import mlflow
            mlflow.tensorflow.autolog()
            mlflow.set_tracking_uri(os.path.join(self.logdir, "../mlruns"))
            mlflow.set_experiment(os.path.basename(self.logdir))

        if self.tensorboard_enabled:
            self.tf_summary_writer = tf.summary.create_file_writer(os.path.join(self.logdir, "summary"))

    def on_train_begin(self, logs=None, *args, **kwargs):
        model = self.model
        self.config["optimizer"] = self.make_dict_json_serializable(model.optimizer.get_config())
        self.config["loss"] = model.loss.get_config() if hasattr(model.loss, "get_config") else model.loss
        self.save_config(self.config, self.save_path)
        #self.save_config(model.get_config(), os.path.join(self.logdir, "model.json")) #causining error for mobilenet

    def on_epoch_end(self, epoch, logs=None):
        if self.current_fold_name not in self.result_log:
            self.result_log[self.current_fold_name] = []
        self.result_log[self.current_fold_name].append({"epoch": epoch, **logs})
        self.save_config(self.result_log, os.path.join(self.logdir, "results.json"))

        if self.tensorboard_enabled:
            for key, val in logs.items():
                tf.summary.scalar(key, val, step=epoch)
            self.tf_summary_writer.flush()

    def append_to_result_log(self, logs):
        self.result_log[self.current_fold_name].append({**logs})
        self.save_config(self.result_log, os.path.join(self.logdir, "results.json"))

    def set_current_fold_name(self, fold_name):
        self.current_fold_name = fold_name

    def get_current_log(self, markdown=False):
        with open(self.save_path) as json_file:
            config_loaded = json.load(json_file)
        if markdown:
            return create_markdown_from_dict(config_loaded)
        return config_loaded

    def update_split_results(self, config, results, name=None):
        if "results" in config:
            for key, val in results.items():
                config['results'][key].append(val)
        else:
            for key, val in results.items():
                if isinstance(val, list):
                    pass
                else:
                    results[key] = [val]
            config.update({"results": results})
        if name is not None:
            self.update_saved_config({name: config})
        else:
            self.update_saved_config(config)

        return config

    def finalize_split_results(self, config, results, name):
        config.update({"final_epoch": results})
        self.update_saved_config({name: config})

    def update_saved_config(self, config):
        if not os.path.exists(self.save_path):
            self.save_config(config, self.save_path)
        else:
            with open(self.save_path, 'r+') as file:
                data = json.load(file)
                data.update(config)
                file.seek(0)
                json.dump(data, file, indent=4)

    def add_model_config(self, config):
        self.update_saved_config({'model': config})

    def add_training_config(self, config):
        self.update_saved_config({'train_config': config})

    def add_generator_config(self, config):
        self.update_saved_config({'generator': config})

    def save_current_config(self):
        self.save_config(self.config, self.save_path)

    @staticmethod
    def save_config(config, filename=None, verbose=False):
        if not isinstance(config, (dict, OrderedDict)):
            raise TypeError('arg config must be a dict or OrderedDict')
        config = OrderedDict(config)

        if filename is None:
            filename = 'config_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '.json'

        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)

        if verbose:
            print('Config has been successfully saved to {}.'.format(filename))
    
    def create_checkpoint_filepath(self):
        class LogString(UserString):
            def format(self, *args, **kwargs):
                log_string = f"epoch{kwargs['epoch']}-"
                log_string += "-".join({"".join((key, f'{val:.2f}')) for key, val in kwargs.items() if "val" in key})
                self = self + "/" + log_string
                return super().format(*args, **kwargs)

        return LogString(self.get_checkpoints_directory())

    def get_checkpoints_directory(self):
        return os.path.join(self.logdir, "checkpoints")

    def get_latest_checkpoint(self):
        return self.find_latest_checkpoint(self.get_checkpoints_directory())

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            potential_checkpoints = [os.path.join(checkpoint_dir, folder) for folder in os.listdir(checkpoint_dir)]
            latest_checkpoint  = max(potential_checkpoints, key=os.path.getctime)
        return latest_checkpoint

    @staticmethod
    def ensure_json_serializable(data):
        """Ensures that the data is JSON serializable. Supports numpy types."""
        if isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8,
                             np.uint16, np.uint32, np.uint64)):
            return int(data)
        elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.ndarray,)):
            return data.tolist()
        else:
            return data

    def make_dict_json_serializable(self, data):
        """Makes a dictionary JSON serializable."""
        return {key: self.ensure_json_serializable(val) for key, val in data.items()}
