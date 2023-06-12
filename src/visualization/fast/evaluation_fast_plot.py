
import fast
import numpy as np
from src.resources.config import *
from src.utils.plot_stations import plot_pred_stations

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

        img_buf = plot_pred_stations(self.labels, classification_arr)

        fast_image = fast.Image.createFromArray(img_buf)
        #new_output_image.setSpacing(classification.getSpacing())

        self.addOutputData(0, fast.Image.createFromArray(fast_image))

class ImageClassificationWindow(object):
    is_running = False

    station_labels = {
        'other': 0,
        '4L': 1,
        '4R': 2,
        '7L': 3,
        '7R': 4,
        '10L': 5,
        '10R': 6,
    }

    def __init__(self, data_path, model_path, model_name, sequence_size=5, framerate=-1):

        # Setup a FAST pipeline
        self.streamer = fast.ImageFileStreamer.create(os.path.join(data_path, 'frame_#.png'), loop=True, framerate=framerate)

        # Neural network model
        self.classification_model = fast.NeuralNetwork.create(os.path.join(model_path, model_name), scaleFactor=1. / 255.)
        self.classification_model.connect(0, self.streamer)

        # Classification (neural network output) to Text
        self.station_classification_plot = ClassificationToPlot.create(name='Station', labels=self.station_labels)
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


def run_nn_image_classification(data_path, model_path, model_name, sequence_size, framerate):

    fast_classification = ImageClassificationWindow(
            data_path=data_path,
            model_path=model_path,
            model_name=model_name,
            sequence_size=sequence_size,
            framerate= framerate
            )

    fast_classification.window.run()


run_nn_image_classification(data_path=local_data_path,
                            model_path=local_model_path,
                            model_name=local_model_name,
                            sequence_size=2,
                            framerate=10
                            )


