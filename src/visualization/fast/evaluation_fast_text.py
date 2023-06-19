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
        print('classification_arr', len(classification_arr))
        print('labels', len(self.labels))
        pred = np.argmax(classification_arr)

        output_text = f'{self.name + ": " if hasattr(self, "name") else "":>16}'

        if self.labels is None:
            output_text += f'{pred:<4}\n'
        else:
            output_text += f'{self.labels[pred]:<40}\n'

        self.addOutputData(0, fast.Text.create(output_text, color=fast.Color.White()))


class ImageClassificationWindow(object):
    is_running = False

    def __init__(self, station_labels, data_path, model_path, model_name, framerate=-1):

        # Setup a FAST pipeline
        self.streamer = fast.ImageFileStreamer.create(filenameFormat=os.path.join(data_path, 'frame_#.png'),
                                                      loop=True, framerate=framerate)
        print('!!!!!!!!!!!!!!!!!!', list(station_labels.keys()))

        # Neural network model
        self.classification_model = fast.ImageClassificationNetwork.create(os.path.join(model_path, model_name),
                                                                           labels=list(station_labels.keys()),
                                                                           scaleFactor=1. / 255.)
        self.classification_model.connect(0, self.streamer)

        # Classification (neural network output) to Text
        #self.station_classification_to_text = ClassificationToText.create(name='Station', labels=list(station_labels.keys()))
        #self.station_classification_to_text.connect(0, self.classification_model)
        self.station_classification_to_text = fast.ClassificationToText.create()
        self.station_classification_to_text.connect(self.classification_model)

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


def run_nn_image_classification(station_labels, data_path, model_path, model_name, framerate):

    fast_classification = ImageClassificationWindow(
            station_labels=station_labels,
            data_path=data_path,
            model_path=model_path,
            model_name=model_name,
            framerate= framerate
            )

    fast_classification.window.run()

#TODO: remove later
def set_stations_config(station_config_nr):
    if station_config_nr == 1:
        stations_config = {
            'other': 0,
            '4L': 1,
            '4R': 2,
            # 'other': 3,
        }
    elif station_config_nr == 2:
        stations_config = {
            'other': 0,
            '4L': 1,
            '4R': 2,
            '7L': 3,
            '7R': 4,
        }
    elif station_config_nr == 3:
        stations_config = {
            'other': 0,
            '4L': 1,
            '4R': 2,
            '7L': 3,
            '7R': 4,
            '10L': 5,
            '10R': 6,
        }
    elif station_config_nr == 4:
        stations_config = {
            'other': 0,
            '4L': 1,
            '4R': 2,
            '7L': 3,
            '7R': 4,
            '10L': 5,
            '10R': 6,
            '11L': 7,
            '11R': 8,
            '7': 9,
        }
    else:
        print("Choose one of the predefined sets of stations: config_nbr={1, 2, 3, 4}")
        exit(-1)

    return stations_config

run_nn_image_classification(station_labels=set_stations_config(station_config_nr),
                            data_path=local_data_path,
                            model_path=local_model_path,
                            model_name=local_model_name,
                            framerate=3
                            )

