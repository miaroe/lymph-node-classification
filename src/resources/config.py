from datetime import datetime
import os

# -----------------------------  TRAINING PARAMETERS ----------------------------------

os.environ['CUDA_VISIBLE_DEVICES']  = "2" # whether to use GPU for training (-1 == no GPU, else GPU)
os.environ['CUDA_DEVICE_ORDER']     = "PCI_BUS_ID"

perform_training    = False
stratified_cv       = False # only accessed if perform_training = True
epochs              = 100
batch_size          = 32
patience            = 20
tf_dataset          = True
augment_data        = True
filter_data         = False
img_size            = 256
station_config_nr   = 1  # class configuration (gives mapping and mapped labels)
model_arch          = "vgg16"  # which architecture/CNN to use - see models.py for info about archs
split_by            = 'stations'
loss                = 'maskedCategoricalCrossEntropy'
mask_poor           = False #TODO: implement mask from config in pipeline (only placed here now to be included in config)
mask_other          = True #TODO: implement mask from config in pipeline (only placed here now to be included in config)

test_split          = 0.0
validation_split    = 0.2
instance_size       = (img_size, img_size, 3)  # Default: (299, 299, 1). Set this to (299, 299, 1) to not downsample further.
learning_rate       = 1e-4  # relevant for the optimizer, Adam used by default (with default lr=1e-3), I normally use 1e-4 when finetuning

date = datetime.today().strftime('%Y-%m-%d')
time = datetime.today().strftime('%H:%M:%S')

# -----------------------------  EVALUATION PARAMETERS ----------------------------------

# use pre-trained model by changing perform_training to False
# and setting the date and time of the model you want to use
if perform_training == False:
    date            = '2023-07-04'
    time            = '14:58:42'

learning_curve          = True
conf_matrix             = True
model_layout            = True
station_distribution    = True
compare_metrics         = True


# -----------------------------  PATHS ----------------------------------
#data_path       = '/mnt/EncryptedData1/LungNavigation/EBUS/annotations/EBUS_Levanger_export_20230420/OE'
db_path         = '/mnt/EncryptedData1/LungNavigation/EBUS/'
data_path       = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger'
model_path      = '/home/miaroe/workspace/lymph-node-classification/output/models/'     + date + '/' + time + '/'
log_path        = '/home/miaroe/workspace/lymph-node-classification/output/logs/'       + date + '/' + time + '/'
reports_path    = '/home/miaroe/workspace/lymph-node-classification/reports/'           + date + '/' + time + '/'

local_data_path     = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/data/Station_7R_001'
local_model_path    = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/models/'
local_model_name    = 'best_model'
local_logfile_path  = '/Users/miarodde/Documents/sintef/ebus-ai/lymph-node-classification/output/logs'


