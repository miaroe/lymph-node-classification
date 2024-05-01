import os
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf

from mlmia.architectures import UNet
from mlmia.keras.losses import DiceLoss
from skimage.transform import resize
from tensorflow import one_hot
from skimage.morphology.convex_hull import grid_points_in_poly
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# modified code from https://github.com/ingridtv/ebus-lymph-node/blob/main/scripts/publication/plot_gt_pred_comparison.py

def create_seg_masks(old_data_path, data_path, seg_model_path):
    """ The function creating segmentation masks for the dataset """

    # delete the old data path if it exists
    if os.path.exists(data_path):
        os.system('rm -r ' + data_path)

    seg_model = load_seg_model(seg_model_path, (256, 256, 1))
    # go through the dataset and for each image create a mask then save it to the new dataset path as a hdf5 file with
    # the same name as the original image
    for ds in os.listdir(old_data_path):
        if os.path.isdir(os.path.join(old_data_path, ds)):
            print('dataset: ', ds)
            for patient in os.listdir(os.path.join(old_data_path, ds)):
                if os.path.isdir(os.path.join(old_data_path, ds, patient)):
                    for station in os.listdir(os.path.join(old_data_path, ds, patient)):
                        if os.path.isdir(os.path.join(old_data_path, ds, patient, station)):
                            for frame in os.listdir(os.path.join(old_data_path, ds, patient, station)):
                                if frame.endswith('.png'):
                                    img_path = os.path.join(old_data_path, ds, patient, station, frame)
                                    img = preprocess_img(img_path)
                                    pred = seg_model.predict(img)[0] # (img_height, img_width, num_classes)

                                    #pred_image = np.argmax(pred, axis=-1) # convert class probability map into single segmentation mask with index of highest probability
                                    #pred_image = np.asarray(one_hot(pred_image, depth=3)) # convert to one-hot encoding
                                    #plot_seg_image(img[0], pred_image)

                                    # saves image and mask to the new dataset path as hdf5 file
                                    new_img_path = os.path.join(data_path, ds, patient, station)
                                    if not os.path.exists(new_img_path):
                                        os.makedirs(new_img_path)
                                    frame = frame.split('.')[0] + '.hdf5'
                                    with h5py.File(os.path.join(new_img_path, frame), 'w') as f:
                                        f.create_dataset('image', data=img[0], dtype='float32')
                                        f.create_dataset('mask', data=pred , dtype='float32')
                                    print('frame: ', frame, ' done')


def label_mapping():
    seg_labels = {0: 'background',
                  1: 'lymph_node',
                  2: 'vessel'}
    return seg_labels


def load_seg_model(seg_model_path, instance_size):
    model = UNet(input_shape=instance_size, classes=3, batch_norm=True)
    model.compile(optimizer='adam', loss=DiceLoss())
    model.load_weights(filepath=os.path.join(seg_model_path, 'best_model')).expect_partial()
    return model

def preprocess_img(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = resize(img, output_shape=(256, 256), preserve_range=True, anti_aliasing=False)
    img = (img[..., None]/255.0).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def ultrasound_sector_mask(reference_image_shape, origin=None, sector_angle=None):

    if origin is None:
        origin = (-60, 1100)  #(-60, (525+1685)/2)  # (y, x) coordinates in pixels
    #plt.scatter(origin[1], origin[0], s=5)
    if sector_angle is None:
        sector_angle = np.pi/3  # radians, pi/3 = 60deg

    # Points in clockwise order from top left
    # Top of sector
    p1 = (100, 1000)  # p1 = (75, 1010)
    p2 = (100, 1100)
    p3 = (100, 1190)  # p3 = (75, 1183)
    # Bottom of sector
    p4 = (820, 1658)
    p5 = (1035, 1658)
    p6 = (1035, 530)
    p7 = (820, 530)

    pts = (p1, p2, p3, p4, p5, p6, p7)
    conv_hull = grid_points_in_poly(shape=reference_image_shape, verts=pts).astype(dtype=int)

    return conv_hull

def get_img_ultrasound_sector_mask():
    img_mask = ultrasound_sector_mask(reference_image_shape=(1080, 1920))[100:1035, 530:1658]
    return resize(img_mask, (256, 256, 1), preserve_range=True, anti_aliasing=False).astype(dtype=bool)

def plot_seg_image(seg_model_path, img_path):
    seg_model = load_seg_model(seg_model_path, (256, 256, 1))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    preprocessed_img = preprocess_img(img_path)
    pred = seg_model.predict(preprocessed_img)[0]  # (img_height, img_width, num_classes)

    mask = np.argmax(pred, axis=-1) # convert class probability map into single segmentation mask with index of highest probability

    c_invalid = (0, 0, 0)
    colors = [(0.2, 0.2, 0.2),  # dark gray = background
              (0.55, 0.4, 0.85), # purple = lymph nodes
              (0.76, 0.1, 0.05)]  # red   = blood vessels

    label_cmap = LinearSegmentedColormap.from_list('label_map', colors, N=3)
    label_cmap.set_bad(color=c_invalid, alpha=0)  # set invalid (nan) colors to be transparent
    image_cmap = plt.cm.get_cmap('gray')
    image_cmap.set_bad(color=c_invalid)

    # Resize mask to cropped frame size
    mask_resized = resize(mask, (935, 1128), preserve_range=True, order=0).astype(mask.dtype)

    mask_resized = np.ma.masked_less_equal(mask_resized, 0)  # mask out background

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.imshow(img, cmap=image_cmap)
    ax.imshow(mask_resized, cmap=label_cmap, interpolation='nearest', alpha=0.4, vmin=0, vmax=2)

    # Define legend patches
    legend_patches = [
        Patch(color=colors[1], label='Lymph node'),
        Patch(color=colors[2], label='Blood vessel'),
    ]

    # Add the legend to the plot
    ax.legend(handles=legend_patches, loc='upper left', fontsize=16, frameon=True, edgecolor='black')

    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # save image
    #fig_path = '/home/miaroe/workspace/lymph-node-classification/figures/'
    #os.makedirs(fig_path, exist_ok=True)
    #plt.savefig(fig_path + 'segmentation_masks.png', bbox_inches='tight')


#plot_seg_image('/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/segmentation-unet-20230614/','/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/baseline/Levanger_and_StOlavs/test/4L/208.png')