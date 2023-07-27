import fast
import numpy as np
from src.resources.config import *
from src.visualization.predicted_stations import plot_pred_stations

fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT) # Uncomment to show debug info


class ClassificationToPlot(fast.PythonProcessObject):

    def __init__(self, labels=None, name=None):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)

        if labels is not None:
            self.labels = labels
        if name is not None:
            self.name = name

    def execute(self):
        classification = self.getInputData(0)
        classification_arr = np.asarray(classification)
        print('classification_arr', classification_arr)

        img_arr = plot_pred_stations(self.labels, classification_arr)
        fast_image = fast.Image.createFromArray(img_arr)

        self.addOutputData(0, fast_image)


class ImageClassificationWindow(object):
    is_running = False

    def __init__(self, station_labels, data_path, model_path, model_name, framerate):

        # Setup a FAST pipeline
        self.streamer = fast.ImageFileStreamer.create(os.path.join(data_path, 'frame_#.png'), loop=True, framerate=framerate)

        # Neural network model
        self.classification_model = fast.NeuralNetwork.create(os.path.join(model_path, model_name), scaleFactor=1. / 127.5) # range [-1, 1] for mobilenet model
        self.classification_model.connect(0, self.streamer)

        # Classification (neural network output) to Plot
        self.station_classification_plot = ClassificationToPlot.create(name='Station', labels=list(station_labels.keys()))
        self.station_classification_plot.connect(0, self.classification_model, 0)

        # Renderers
        self.image_renderer = fast.ImageRenderer.create().connect(self.streamer)
        self.classification_renderer = fast.ImageRenderer.create()
        self.classification_renderer.connect(self.station_classification_plot)

        # Set up video window
        self.window = fast.DualViewWindow2D.create(
            width=800,
            height=900,
            bgcolor=fast.Color.Black(),
            verticalMode=True  # defaults to False
        )

        self.window.connectTop([self.classification_renderer])
        self.window.addRendererToTopView(self.classification_renderer)
        self.window.connectBottom([self.image_renderer])
        self.window.addRendererToBottomView(self.image_renderer)

        # Set up playback widget
        self.widget = fast.PlaybackWidget(streamer=self.streamer)
        self.window.connect(self.widget)

    def run(self):
        self.window.run()


def run_nn_image_classification(station_labels, data_path, model_path, model_name, framerate):

    fast_classification = ImageClassificationWindow(
            station_labels=station_labels,
            data_path=data_path,
            model_path=model_path,
            model_name=model_name,
            framerate=framerate
            )

    fast_classification.window.run()


run_nn_image_classification(station_labels=get_stations_config(station_config_nr),
                            data_path=local_data_path,
                            model_path=local_model_path,
                            model_name=local_model_name,
                            framerate=1
                            )


