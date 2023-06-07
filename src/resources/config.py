from datetime import datetime
import os

# -----------------------------  TRAINING PARAMETERS ----------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # whether to use GPU for training (-1 == no GPU, else GPU)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

perform_training = True
epochs = 1
batch_size = 32
patience = 20
augment_data = False
img_size = 256
label_config = 3  # class configuration (gives mapping and mapped labels)
model_arch = "resnet"  # which architecture/CNN to use - see models.py for info about archs

test_split = 0.0
validation_split = 0.2
instance_size = (img_size, img_size, 3)  # Default: (299, 299, 1). Set this to (299, 299, 1) to not downsample further.
learning_rate = 1e-4  # relevant for the optimizer, Adam used by default (with default lr=1e-3), I normally use 1e-4 when finetuning
datetime_format = '%Y-%m-%d-%H%M%S'

model_name = f'EBUSClassification' \
                 f'_Stations_config-{label_config}' \
                 f'_{datetime.now().strftime(datetime_format)}' \
                 f'_Epochs-{epochs}' \
                 f'_ImageSize-{img_size}' \
                 f'_BatchSize-{batch_size}' \
                 f'_Augmentation-{augment_data}' \
                 f'_ValPercent-{int(100*validation_split)}'

# -----------------------------  PATHS ----------------------------------

data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Levanger'
save_model_path = "/home/miaroe/workspace/lymph-node-classification/output/models/"
history_path = "/home/miaroe/workspace/lymph-node-classification/output/history/"
logfile_path = "/home/miaroe/workspace/lymph-node-classification/output/logs"