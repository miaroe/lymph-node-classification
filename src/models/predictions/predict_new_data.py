import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import cv2

from src.resources.architectures.ml_models import get_arch
from src.resources.config import get_stations_config, get_num_stations
from src.resources.loss import get_loss
from src.visualization.confusion_matrix import confusion_matrix_and_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.resources.train_config import get_config
from src.utils.get_paths import get_frame_paths, get_test_station_paths

# Used to create a test dataset using data from Ålesund, needs to be cropped differently


def get_label_one_hot(station_folder, stations_config, num_stations):
    label = stations_config[station_folder]
    label_one_hot = tf.keras.utils.to_categorical(label, num_classes=num_stations)
    label_one_hot = tf.cast(label_one_hot, tf.float32)
    return label_one_hot

def get_path_labels(paths, stations_config, num_stations):
    labels = [path.split('/')[-1] for path in paths]
    return [get_label_one_hot(label, stations_config, num_stations) for label in labels]

def preprocess_frames(image_path):
    frame = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=None)
    frame = np.array(frame)
    frame = frame[105:635, 585:1335]
    frame = tf.cast(frame, tf.float32)
    frame = tf.image.resize(frame, [224, 224], method='nearest')
    frame = frame / 127.5 - 1
    return frame

def load_image_sequence(frame_paths, seq_length):
    sequence = [preprocess_frames(frame_path) for frame_path in frame_paths]
    if len(sequence) != seq_length:
        # add zero padding to make the total equal seq_length
        zero_frame = tf.zeros_like(sequence[-1], dtype=tf.float32)
        num_repeats = seq_length - len(frame_paths)
        sequence = sequence + ([zero_frame] * num_repeats)
    sequence = tf.stack(sequence)
    return sequence

def get_predictions(sequence, model):
    sequence = np.array(sequence)  # Convert list to np.array
    sequence = np.expand_dims(sequence, axis=0)  # Model expects batch dimension
    predictions = model.predict(sequence)
    return predictions[0]

def create_sequences_test(station_path, seq_length, convert_from_tensor):
    if convert_from_tensor:
        station_path = station_path.numpy().decode('utf-8')
    frame_paths = get_frame_paths(station_path, 'sequence')  # gets frame paths for one station
    num_frames = len(frame_paths)
    sequence_paths = [frame_paths[i: i + seq_length] for i in range(0, num_frames, seq_length)]
    sequences = [load_image_sequence(sequence, seq_length) for sequence in sequence_paths]
    return sequences

def predict_sequence(model_path, model_name, test_path, seq_length):
    model = tf.keras.models.load_model(os.path.join(model_path, model_name))
    sequence_predictions = []
    true_labels = []
    for patient in os.listdir(test_path):
        patient_path = os.path.join(test_path, patient)
        print('patient_path:', patient_path)
        for station in os.listdir(patient_path):
            station_path = os.path.join(patient_path, station)
            frame_paths = get_frame_paths(station_path, 'sequence')
            num_frames = len(frame_paths)
            sequences = [frame_paths[i: i + seq_length] for i in range(0, num_frames, seq_length)]
            for sequence in sequences:
                loaded_sequence = load_image_sequence(sequence, seq_length)
                prediction = get_predictions(loaded_sequence, model).tolist()
                sequence_predictions.append(np.argmax(prediction))
                true_labels.append(get_stations_config(3)[station.split('_')[1]])

    print('sequence_predictions:', sequence_predictions)
    print('true_labels:', true_labels)
    print('accuracy:', accuracy_score(true_labels, sequence_predictions))
    return sequence_predictions, true_labels

#predict_sequence(model_path='/home/miaroe/workspace/lymph-node-classification/output/models/2024-04-19/22:44:29',
#                 model_name='best_model', test_path='/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Aalesund',
#                 seq_length=10)


# evaluate sequence model per sequence
def evaluate_sequence_model_per_seq(model_path, reports_path, seq_length, stations_config_nr, batch_size):
    print("Evaluating model: " + model_path)

    test_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Aalesund'
    stations_config = get_stations_config(stations_config_nr)
    num_stations = get_num_stations(stations_config_nr)
    test_station_paths = get_test_station_paths(test_path, 'sequence')

    sequences_list = []
    labels_list = []
    for station_path in test_station_paths:
        sequences = create_sequences_test(station_path, seq_length, False)
        print('station_path:', station_path)
        labels = get_path_labels([station_path] * len(sequences), stations_config, num_stations)
        sequences_list.extend(sequences)
        labels_list.extend(labels)

    print('num sequences:', len(sequences_list))
    print('num labels:', len(labels_list))

    test_ds = tf.data.Dataset.from_tensor_slices((sequences_list, labels_list))
    test_ds = test_ds.batch(batch_size)

    # print the value range of the pixels
    for images, labels in test_ds.take(1):
        print('images shape: ', images.shape)  # (batch_size, seq_length, 256, 256, 3)
        print('min:', tf.reduce_min(images))
        print('max:', tf.reduce_max(images))
    num_images = 0
    
    for i, (images, labels) in enumerate(test_ds.take(3)):
        print('images shape: ', images.shape)  # (1, seq_length, 256, 256, 3)
        print('labels shape: ', labels.shape)  # (1, 8)
        for seq in range(images.shape[0]):
            plt.figure(figsize=(10, 10))
            for image in range(seq_length):
                num_images += 1
                plt.subplot(4, 4, image + 1)
                #image_norm = images[seq][image]
                image_norm = (images[seq][image] + 1) * 127.5
                #image_norm = (images[seq][image]) * 255.0
                plt.imshow(image_norm.numpy().astype("uint8"))
                plt.title(f"Frame {image}, Label: {np.argmax(labels.numpy()[seq])}",
                          fontsize=10)
                plt.axis("off")
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.suptitle(f"Batch {i}, Sequence {seq}", fontsize=16)
            plt.show()
    print(f"Total images: {num_images}")


    config_path = os.path.join(model_path, 'config.json')
    config = get_config(config_path)
    train_config = config["train_config"]

    model = get_arch(train_config.get('model_arch'), train_config.get('instance_size'),
                     train_config.get('num_stations'), stateful=False) # not stateful here, batches are shuffled
    model.compile(loss=get_loss(train_config.get('loss')), optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    model.load_weights(filepath=os.path.join(model_path, 'best_model')).expect_partial()

    #stateless_model = tf.keras.models.load_model(os.path.join(model_path, 'best_model'))
    #model = create_model(instance_size=(224, 224, 3), num_stations=8, stateful=False)
    #model.set_weights(stateless_model.get_weights())

    score_test = model.evaluate(test_ds, return_dict=True, steps=None)
    with open(os.path.join(reports_path, 'test_metrics.txt'), 'a') as f:
        f.write(f'\n\nTest metrics for model: {model_path}\n')
        f.write(f'{"Metric":<12}{"Value"}\n')
        for metric, value in score_test.items():
            f.write(f'{metric:<12}{value:<.4f}\n')

    #debug
    # Initialize the true labels and predicted labels arrays
    true_labels = []
    pred_labels = []

    for batch in test_ds:
        images, labels = batch  # shape=(4, seq_length, 224, 224, 3)
        pred_probs = model.predict(images)
        batch_pred_labels = np.argmax(pred_probs, axis=1)
        true_labels.extend(np.argmax(labels, axis=1))
        pred_labels.extend(batch_pred_labels)
    print('true_labels:', true_labels)
    print('pred_labels:', pred_labels)
    print('accuracy:', accuracy_score(true_labels, pred_labels))
    print('precision:', precision_score(true_labels, pred_labels, average='weighted'))
    print('recall:', recall_score(true_labels, pred_labels, average='weighted'))

    confusion_matrix_and_report(true_labels, pred_labels, num_stations,
                                train_config.get('stations_config'),
                                reports_path, 'sequence_test_Aalesund')


#evaluate_sequence_model_per_seq(model_path='/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-09/12:07:18',
#                         reports_path='/home/miaroe/workspace/lymph-node-classification/reports/2024-05-09/12:07:18/',
#                         seq_length=10, stations_config_nr=3, batch_size=1)



'''
/home/miaroe/workspace/lymph-node-classification/output/models/2024-03-03/22:33:14
accuracy: 0.39344262295081966
precision: 0.5774274905422446
recall: 0.39344262295081966

/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-02/16:08:35
accuracy: 0.32786885245901637
precision: 0.5446899501069138
recall: 0.32786885245901637

/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-05/21:10:47
accuracy: 0.19672131147540983
precision: 0.30700447093889716
recall: 0.19672131147540983

/home/miaroe/workspace/lymph-node-classification/output/models/2024-05-05/21:11:14
accuracy: 0.29508196721311475
precision: 0.4458756180067655
recall: 0.29508196721311475
'''

# plot image

#image_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/EBUS_Aalesund/Patient_20240419-102838/4L/frame_1314.png'
image_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/FullVideos/EBUS_StOlavs_full_videos/Patient_20240502-084504/Sequence_001/frame_2349.png'
#image_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/test/EBUS_Levanger_Patient_022/4L/frame_1693.png'
#image_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/FullVideos/EBUS_StOlavs_full_videos/Patient_001/Sequence_001/frame_2083.png'
image = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(1080, 2040)) # (height, width) 2048x1080
image = np.array(image)
image = image[100:1035, 510:1845]  #[y1:y2, x1:x2]
#image = image[100:1035, 530:1658]
print('image shape:', image.shape)
#image = image[150:775, 450:1430] 625 845 1.352
image = tf.cast(image, tf.float32)
image = tf.image.resize(image, [224, 224], method='nearest')

#image = cv2.imread(image_path)
#image = cv2.resize(image, (1280, 1024))
#print('image shape:', image.shape)

#fig = plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)

plt.imshow(image.numpy().astype("uint8"))
plt.show()







# [105:635, 585:1335] accuracy: 0.3220338983050847, precision: 0.5832819722650231, recall: 0.3220338983050847 (using tf.keras.utils.load_img)


# with the whole US sector: [110:725, 515:1400] accuracy: 0.0847457627118644, precision: 0.2919020715630885, recall: 0.0847457627118644
# cropped like training dataset using lines: [110:670, 600:1320] accuracy: 0.2033898305084746, precision: 0.4133239171374765, recall: 0.2033898305084746
# cropped like training dataset manually: [110:630, 580:1330] accuracy: 0.2711864406779661, precision: 0.5067124024751143, recall: 0.2711864406779661
# cropped like training dataset manually: [100:630, 580:1330] accuracy: 0.288135593220339, precision: 0.6246973365617433, recall: 0.288135593220339

# [100:630, 580:1330] accuracy: 0.288135593220339, precision: 0.6246973365617433, recall: 0.288135593220339
# [90:620, 580:1330] accuracy: 0.2033898305084746, precision: 0.5414043583535109, recall: 0.2033898305084746
# [110:650, 580:1330] accuracy: 0.23728813559322035, precision: 0.4309430682312039, recall: 0.23728813559322035
# [100:630, 570:1340] accuracy: 0.2711864406779661, precision: 0.5883777239709443, recall: 0.2711864406779661
# [100:630, 590:1320] accuracy: 0.2542372881355932, precision: 0.575991804805364, recall: 0.2542372881355932
# [100:630, 600:1310] accuracy: 0.288135593220339, precision: 0.622547508988187, recall: 0.288135593220339
# [120:600, 580:1330] accuracy: 0.2711864406779661, precision: 0.5654358353510895, recall: 0.2711864406779661
# [90:640, 580:1330] accuracy: 0.22033898305084745, precision: 0.5677966101694916, recall: 0.22033898305084745
# [100:630, 570:1340] accuracy: 0.2711864406779661, precision: 0.5883777239709443, recall: 0.2711864406779661
# [95:625, 575:1325] accuracy: 0.23728813559322035, precision: 0.5609879762422135, recall: 0.23728813559322035
# [105:635, 585:1335] accuracy: 0.3050847457627119, precision: 0.5440677966101695, recall: 0.3050847457627119
