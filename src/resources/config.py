from datetime import datetime
import os

# -----------------------------  TRAINING PARAMETERS ----------------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = "0" # whether to use GPU for training (-1 == no GPU, else GPU)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

perform_training = True
stratified_cv = False # only accessed if perform_training = True
epochs = 50
batch_size = 32
patience = 20
tf_dataset = True
augment_data = True
img_size = 256
station_config_nr = 1  # class configuration (gives mapping and mapped labels)
model_arch = "vgg16"  # which architecture/CNN to use - see models.py for info about archs
split_by = 'stations'
loss = 'focalLoss'

test_split = 0.0
validation_split = 0.2
instance_size = (img_size, img_size, 3)  # Default: (299, 299, 1). Set this to (299, 299, 1) to not downsample further.
learning_rate = 1e-4  # relevant for the optimizer, Adam used by default (with default lr=1e-3), I normally use 1e-4 when finetuning
date = datetime.today().strftime('%Y-%m-%d')
datetime_format = '%Y-%m-%d_%H:%M:%S'
# -----------------------------  EVALUATION PARAMETERS ----------------------------------

learning_curve = True
conf_matrix =False
model_layout = False
station_distribution = False
compare_metrics = True

model_name = f'Arch-{model_arch}' \
                 f'_{datetime.now().strftime(datetime_format)}' \
                 f'_Stations_config_nr-{station_config_nr}' \
                 f'_Epochs-{epochs}' \
                 f'_Loss-{loss}' \
                 f'_ImageSize-{img_size}' \
                 f'_BatchSize-{batch_size}' \
                 f'_Augmentation-{augment_data}' \
                 f'_ValPercent-{int(100*validation_split)}'


# -----------------------------  PATHS ----------------------------------

data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger'
model_path = '/home/miaroe/workspace/lymph-node-classification/output/models/' + date + '/'
history_path = '/home/miaroe/workspace/lymph-node-classification/output/history/' + date + '/'
logfile_path = '/home/miaroe/workspace/lymph-node-classification/output/logs/' + date + '/'
reports_path = '/home/miaroe/workspace/lymph-node-classification/reports/' + date + '/'

local_data_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/data/Station_4L_001'
local_model_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/models/'
local_model_name = 'Arch-inception_Stations_config_nr-4_2023-06-16_12:45:02_Epochs-100_ImageSize-256_BatchSize-32_Augmentation-True_ValPercent-20'
local_logfile_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/logs'
local_history_path = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/history'
