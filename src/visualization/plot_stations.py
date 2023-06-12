import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib.ticker as mtick

plt.style.use('dark_background')


def plot_pred_stations(station_labels, pred):
    fig = plt.figure(figsize=(10, 5))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    pred_max_i = np.argsort(pred)[::-1][:3] #index of top three predictions

    for i in pred_max_i:
        plt.bar(list(station_labels.keys())[i], pred[i])

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    #save img to numpy array
    img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img_arr

'''
pred = [0.04027496, 0.01737576, 0.0540364, 0.02898372, 0.810823, 0.02736586, 0.02114019]
plot_pred_stations(station_labels, pred)
'''