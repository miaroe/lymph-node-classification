import tensorflow as tf
import os
import matplotlib.pyplot as plt

from src.resources.augmentation import GammaTransform, ContrastScale, Blur, BrightnessTransform, GaussianShadow, \
    RandomAugmentation, Rotation, NonLinearMap, RandomAugmentationSequence
"""
Functions to visualize the effect of different augmentations on an image
"""

data_path = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/val/EBUS_Levanger_Patient_022'
data_path_image = '/mnt/EncryptedData1/LungNavigation/EBUS/ultrasound/sequence/Levanger_and_StOlavs/val/EBUS_Levanger_Patient_022/4L/frame_1690.png'

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [256, 256], method='nearest')
    img = img / 255.0  # Normalize the image to [0, 1]
    return img

# Scan data_path for folders and select one image from each
images = []
labels = []
for label_folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, label_folder)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                images.append(load_image(image_path))
                labels.append(label_folder)
                break  # Select only one image per folder
def data_augmentation(x):
    # augmentation_layers = [GammaTransform(low=0.5, high=1.5), Blur(sigma_max=1.0),
    #                       Rotation(max_angle=30), ContrastScale(min_scale=0.5, max_scale=1.5),
    #                       GaussianShadow(sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.9), strength=(0.5, 0.8))]
    augmentation_layers = [GammaTransform(low=0.5, high=1.5), Blur(sigma_max=1.0),
                           Rotation(max_angle=30), ContrastScale(min_scale=0.5, max_scale=1.5),
                           GaussianShadow(sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.9), strength=(0.5, 0.8))]
    return RandomAugmentationSequence(augmentation_layers)(x)


def test_aug():
    #images = tf.stack(images)
    #augmented_images = data_augmentation(images)
    augmented_images = []
    for image in images:
        print('new image')
        augmented_images.append(data_augmentation(image))

    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.rcParams['axes.grid'] = False
    for i in range(len(images)):
        plt.subplot(1, 2, 1)
        plt.title('Original image', fontsize=20)
        plt.imshow(images[i])
        plt.subplot(1, 2, 2)
        plt.title('Augmented image', fontsize=20)
        plt.imshow(augmented_images[i])
        plt.suptitle(labels[i], fontsize=20)
        plt.show()


def save_aug_images():
    image = load_image(image_path)
    augmentation_layers = [GammaTransform(low=0.5, high=1.5), Blur(sigma_max=1.0),
                           Rotation(max_angle=30), ContrastScale(min_scale=0.5, max_scale=1.5),
                           GaussianShadow(sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.9), strength=(0.5, 0.8))]
    augmentation_layers_names = ['GammaTransform', 'Blur', 'Rotation', 'ContrastScale', 'GaussianShadow']

    fig_path = os.path.join('/home/miaroe/workspace/lymph-node-classification/figures', 'aug/')
    os.makedirs(fig_path, exist_ok=True)

    plt.style.use('dark_background')
    plt.axis('off')

    plt.imshow(image)
    plt.savefig(fig_path + 'original.png', bbox_inches='tight', format='png', dpi=300)
    for layer, layer_name in zip(augmentation_layers, augmentation_layers_names):
        augmented_image = RandomAugmentation([layer])(image)
        plt.imshow(augmented_image)
        plt.savefig(fig_path + layer_name + '.png', bbox_inches='tight', format='png', dpi=300)



save_aug_images()







