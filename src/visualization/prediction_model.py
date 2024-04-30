import matplotlib.pyplot as plt
import os
import numpy as np

def plot_predictions(model, test_ds, model_type, station_names, reports_path):

    # plot the first 9 images in the test set, and their predicted labels and true labels
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 10))
    for images, labels in test_ds.take(1):
        print('images.shape: ', images.shape)
        # predictions = get_predictions(model, images)
        predictions = model.predict(images)

        # print min and max of image[0]
        print('min: ', np.min(images[0]))
        print('max: ', np.max(images[0]))
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            if model_type == 'baseline':
                image = ((images[i] + 1.) / 2.) * 255 # denormalize image
                #image = images[i] * 255
            else:  # model_type == 'sequence' and has an extra dimension
                image = images[0][i]
                # normalize image
                image = image * 255
            plt.imshow(np.array(image).astype("uint8"))

            if len(station_names) > 2:
                plt.title("Predicted: {}, True: {}".format(station_names[np.argmax(predictions[i])],
                                                           station_names[np.argmax(labels[i])]))
            else:
                plt.title("Predicted: {}, True: {}".format(station_names[int(predictions[i][0])],
                                                           station_names[labels.numpy()[i]]))
            plt.axis("off")

    # save figure
    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + 'predictions.png')
