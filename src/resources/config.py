from datetime import datetime
import os

# -----------------------------  PATHS ----------------------------------

data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger'
model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/'
history_path = '/home/miaroe/workspace/lymph-node-classification/output/history/'
logfile_path = '/home/miaroe/workspace/lymph-node-classification/output/logs'
reports_path = '/home/miaroe/workspace/lymph-node-classification/reports'

local_data_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/data/Station_4R_001'
local_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/models/'
local_model_name = 'EBUSClassification_Stations_-10_2023-06-12-143330_Epochs-10_ImageSize-256_BatchSize-32_Augmentation-False_ValPercent-20'
local_logfile_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/logs'
local_history_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/history'


# -----------------------------  STATIONS CONFIG ----------------------------------

# Label lymph nodes in ascending order, and L before R stations
#   e.g. 4L, 4R, 7L, 7R, 10L, 10R, 11L, 11R, ...

def set_stations_config(station_config_nr):
    if station_config_nr == 1:
        stations_config = {
            'other': 0,
            '4L': 1,
            '4R': 2,
            # 'other': 3,
        }
    elif station_config_nr == 2:
        stations_config = {
            'other': 0,
            '4L': 1,
            '4R': 2,
            '7L': 3,
            '7R': 4,
        }
    elif station_config_nr == 3:
        stations_config = {
            'other': 0,
            '4L': 1,
            '4R': 2,
            '7L': 3,
            '7R': 4,
            '10L': 5,
            '10R': 6,
        }
    elif station_config_nr == 4:
        stations_config = {
            'other': 0,
            '4L': 1,
            '4R': 2,
            '7L': 3,
            '7R': 4,
            '10L': 5,
            '10R': 6,
            '11L': 7,
            '11R': 8,
            '7': 9,
        }
    else:
        print("Choose one of the predefined sets of stations: config_nbr={1, 2, 3, 4}")
        exit(-1)

    return stations_config

def get_num_stations(station_config_nr):
    stations_config = set_stations_config(station_config_nr)
    return len(stations_config.keys())


# -----------------------------  TRAINING PARAMETERS ----------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # whether to use GPU for training (-1 == no GPU, else GPU)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

perform_training = False
epochs = 10
batch_size = 32
patience = 20
samples_per_load=10
tf_dataset=True
augment_data = False
img_size = 256
station_config_nr = 4  # class configuration (gives mapping and mapped labels)
model_arch = "mobilenet"  # which architecture/CNN to use - see models.py for info about archs
split_by = 'stations'

test_split = 0.0
validation_split = 0.2
instance_size = (img_size, img_size, 3)  # Default: (299, 299, 1). Set this to (299, 299, 1) to not downsample further.
learning_rate = 1e-4  # relevant for the optimizer, Adam used by default (with default lr=1e-3), I normally use 1e-4 when finetuning
datetime_format = '%Y-%m-%d-%H%M%S'


# -----------------------------  EVALUATION PARAMETERS ----------------------------------
conf_matrix = True
model_layout = True


model_name = f'EBUSClassification-{model_arch}' \
                 f'_Stations-{get_num_stations(station_config_nr)}' \
                 f'_{datetime.now().strftime(datetime_format)}' \
                 f'_Epochs-{epochs}' \
                 f'_ImageSize-{img_size}' \
                 f'_BatchSize-{batch_size}' \
                 f'_Augmentation-{augment_data}' \
                 f'_ValPercent-{int(100*validation_split)}'