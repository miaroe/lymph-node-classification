from datetime import datetime
import os


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


def get_num_stations(station_config_nr):
    return len(get_stations_config(station_config_nr).keys())


# -----------------------------  TRAINING PARAMETERS ----------------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # whether to use GPU for training (-1 == no GPU, else GPU)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

perform_training = True
model_type = 'sequence'  # baseline or sequence
epochs = 100
steps_per_epoch = 200
validation_steps = 50
stride = 4
batch_size = 8
patience = 20
filter_data = False # not used atm
augment = True
img_size = 224
station_config_nr = 3  # class configuration (gives mapping and mapped labels)
stations_config = get_stations_config(station_config_nr)
num_stations = get_num_stations(station_config_nr)
model_arch = "cnn-lstm"  # which architecture/CNN to use - see models.py for info about archs
loss = 'categoricalCrossEntropy' # binaryCrossEntropy for binary, categoricalCrossEntropy / focalCrossEntropy for multiclass
mask_poor = False

stratified_cv = False
test_split = 0.1
validation_split = 0.2
instance_size = (img_size, img_size, 3)  # Default: (299, 299, 1). Set this to (299, 299, 1) to not downsample further.
learning_rate = 0.0001  # relevant for the optimizer, Adam used by default (with default lr=1e-3), I normally use 1e-4 when finetuning
seq_length = 5  # number of frames in each sequence

date = datetime.today().strftime('%Y-%m-%d')
time = datetime.today().strftime('%H:%M:%S')

# -----------------------------  EVALUATION PARAMETERS ----------------------------------
visualize_predictions = False
learning_curve = True
conf_matrix = True
model_layout = False
station_distribution = True
compare_metrics = True

# -----------------------------  PATHS ----------------------------------
if model_type == 'baseline':
    data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger_baseline'
    test_ds_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger_test/baseline'
elif model_type == 'sequence':
    data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger_sequence'
    test_ds_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger_test/sequence'

db_path = '/mnt/EncryptedData1/LungNavigation/EBUS/'
model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/' + date + '/' + time + '/'
log_path = '/home/miaroe/workspace/lymph-node-classification/output/logs/' + date + '/' + time + '/'
reports_path = '/home/miaroe/workspace/lymph-node-classification/reports/' + date + '/' + time + '/'

local_data_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/data/Station_7R_001'
local_full_video_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/data/Patient_037/Sequence_001'
local_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/models/'
local_model_name = 'best_model'
local_logfile_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/logs'
