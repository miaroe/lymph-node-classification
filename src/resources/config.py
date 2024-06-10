from datetime import datetime
import os

from src.utils.get_paths import count_number_of_training_samples


# -----------------------------  STATIONS CONFIG ----------------------------------

# Label lymph nodes in ascending order, and L before R stations
#   e.g. 4L, 4R, 7L, 7R, 10L, 10R, 11L, 11R, ...

def get_stations_config(station_config_nr):
    if station_config_nr == 1:  # binary classification
        return {
            '4L': 0,
            '7R': 1,
        }
    elif station_config_nr == 2:  # multiclass classification with almost balanced classes
        return {
            '4L': 0,
            '4R': 1,
            '7L': 2,
            '7R': 3,
            '10R': 4,
            '11R': 5
        }

    elif station_config_nr == 3:  # multiclass classification with unbalanced classes, deleted 7
        return {
            '4L': 0,
            '4R': 1,
            '7L': 2,
            '7R': 3,
            '10L': 4,
            '10R': 5,
            '11L': 6,
            '11R': 7
        }
    elif station_config_nr == 4:  # multiclass classification with unbalanced classes, deleted 7 and 10L
        return {
            '4L': 0,
            '4R': 1,
            '7L': 2,
            '7R': 3,
            '10R': 4,
            '11L': 5,
            '11R': 6
        }
    elif station_config_nr == 5:  # multiclass classification with unbalanced classes, combined 7, 7R and 7L
        return {
            '4L': 0,
            '4R': 1,
            '7': 2,
            '10L': 3,
            '10R': 4,
            '11L': 5,
            '11R': 6
        }
    elif station_config_nr == 6:  # multiclass classification with unbalanced classes, combined 7, 7R and 7L, removed 10L
        return {
            '4L': 0,
            '4R': 1,
            '7': 2,
            '10R': 3,
            '11L': 4,
            '11R': 5
        }
    elif station_config_nr == 7:  # multiclass classification with station 0
        return {
            '0': 0,
            '4L': 1,
            '4R': 2,
            '7L': 3,
            '7R': 4,
            '10L': 5,
            '10R': 6,
            '11L': 7,
            '11R': 8
        }


def get_num_stations(station_config_nr):
    return len(get_stations_config(station_config_nr).keys())

# -----------------------------  MODEL TYPE ----------------------------------
model_type = 'baseline'  # baseline, sequence or sequence_with_segmentation


# -----------------------------  PATHS ----------------------------------

date = datetime.today().strftime('%Y-%m-%d')
time = datetime.today().strftime('%H:%M:%S')

if model_type == 'baseline':
    data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/baseline/Levanger_and_StOlavs'
elif model_type == 'sequence':
    data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs'
elif model_type == 'sequence_cv':
    data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence_cv'
elif model_type == 'sequence_with_segmentation':
    old_data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs'
    data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence_segmentation/Levanger_and_StOlavs'
    seg_model_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/segmentation-unet-20230614/'
else:
    raise ValueError('Model type not recognized')

full_video_path = ['/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/FullVideos/EBUS_Levanger_full_videos',
                   '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/FullVideos/EBUS_Levanger_with_consent_full_videos',
                   '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/FullVideos/EBUS_StOlavs_full_videos']
db_path = '/mnt/EncryptedData1/LungNavigation/EBUS/'
model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/' + date + '/' + time + '/'
log_path = '/home/miaroe/workspace/lymph-node-classification/output/logs/' + date + '/' + time + '/'
reports_path = '/home/miaroe/workspace/lymph-node-classification/reports/' + date + '/' + time + '/'

local_data_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/data/Station_7R_001'
local_full_video_path = '/Users/miarodde/Documents/sintef/ebus-ai/EBUS_Levanger_full_videos/Patient_036/Sequence_001'
local_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/models/'
local_model_name = 'best_model'
local_logfile_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/logs'



# -----------------------------  TRAINING PARAMETERS ----------------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = "2"  # whether to use GPU for training (-1 == no GPU, else GPU)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

perform_training = True
perform_segmentation = False

station_config_nr = 3  # class configuration (gives mapping and mapped labels)
stations_config = get_stations_config(station_config_nr)
num_stations = get_num_stations(station_config_nr)
num_train, num_val = count_number_of_training_samples(data_path, fold=None, model_type=model_type) # change fold to None if not using cv
filter_data = False # only includes sequences labeled as 'good quality'

img_size = 224
instance_size = (img_size, img_size, 3)
augment = True
epochs = 1
batch_size = 32
patience = 20
model_arch = "mobileNetV3Small"  # which architecture/CNN to use - see models.py for info about archs
loss = 'categoricalCrossEntropy' # binaryCrossEntropy for binary, categoricalCrossEntropy / focalCrossEntropy for multiclass
learning_rate = 0.0001
stratified_cv = False
test_split = 0.1
validation_split = 0.2

use_quality_weights = False # whether to use sample_weights given by the quality in the loss function

# for sequence model
steps_per_epoch = num_train // batch_size
validation_steps = num_val // batch_size
set_stride = True # whether to use random stride (no. between 1 and 3) or not (always 1)
seq_length = 10  # number of frames in each sequence, not relevent when using full video
full_video = False # whether to create sequences of full station video or not
use_gen = False  # whether to use generator or not

# -----------------------------  EVALUATION PARAMETERS ----------------------------------
visualize_predictions = True
learning_curve = True
conf_matrix = True
model_layout = False
station_distribution = True
compare_metrics = True