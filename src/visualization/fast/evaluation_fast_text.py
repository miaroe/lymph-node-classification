import fast # Must import FAST before rest of pyside2
import numpy as np
from src.resources.config import *

fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT) # Uncomment to show debug info


class ClassificationToText(fast.PythonProcessObject):

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
        pred = np.argmax(classification_arr)

        output_text = f'{self.name}'

        if self.labels is None:
            output_text += f'{pred:<4}\n'
        else:
            output_text += f'{list(self.labels.keys())[pred]:<40}\n'


        self.addOutputData(0, fast.Text.create(output_text, color=fast.Color.White()))


class ImageClassificationWindow(object):
    is_running = False

    def __init__(self, station_labels, data_path, model_path, model_name, sequence_size=5, framerate=-1):

        # Setup a FAST pipeline
        self.streamer = fast.ImageFileStreamer.create(os.path.join(data_path, 'frame_#.png'), loop=True, framerate=framerate)

        # Neural network model
        self.classification_model = fast.NeuralNetwork.create(os.path.join(model_path, model_name), scaleFactor=1. / 255.)
        self.classification_model.connect(0, self.streamer)

        # Classification (neural network output) to Text
        self.station_classification_to_text = ClassificationToText.create(name='Station', labels=station_labels)
        self.station_classification_to_text.connect(0, self.classification_model, 0)

        # Renderers
        self.image_renderer = fast.ImageRenderer.create().connect(self.streamer)
        self.classification_renderer = fast.TextRenderer.create(fontSize=48)
        self.classification_renderer.connect(self.station_classification_to_text)

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


def run_nn_image_classification(station_labels, data_path, model_path, model_name, sequence_size, framerate):

    fast_classification = ImageClassificationWindow(
            station_labels=station_labels,
            data_path=data_path,
            model_path=model_path,
            model_name=model_name,
            sequence_size=sequence_size,
            framerate= framerate
            )

    fast_classification.window.run()


run_nn_image_classification(station_labels=set_stations_config(station_config_nr),
                            data_path=local_data_path,
                            model_path=local_model_path,
                            model_name=local_model_name,
                            sequence_size=2,
                            framerate=10
                            )


