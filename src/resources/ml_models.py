import tensorflow as tf
from mlmia.architectures import CVCNet
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape, LSTM, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import PReLU


# see here for already built-in pretrained architectures:
# https://keras.io/api/applications/

def get_arch(model_arch, instance_size, num_stations, seq_length=None, stateful=False):
    # basic
    if model_arch == "basic":
        # define model (some naive network)
        model = Sequential()  # example of creation of TF-Keras model using the Sequential
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=instance_size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(num_stations, activation='softmax'))

    elif model_arch == "inception":
        # InceptionV3 (typical example arch) - personal preference for CNN classification (however, quite expensive and might be overkill in a lot of scenarios)
        some_input = Input(shape=instance_size)
        base_model = InceptionV3(include_top=False, weights="imagenet", pooling=None, input_tensor=some_input)
        base_model.trainable = False
        x = base_model.output
        x = Flatten()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Activation('relu')(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)  # example of creation of TF-Keras model using the functional API

    elif model_arch == "resnet":
        # ResNet-50, another very popular arch, can be done similarly as for InceptionV3 above
        some_input = Input(shape=instance_size)
        base_model = ResNet50(include_top=False, weights="imagenet", pooling=None, input_tensor=some_input)
        base_model.trainable = False
        x = base_model.output
        x = Flatten()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Activation('relu')(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)

    elif model_arch == "inception_multi-task":
        # Example of a multi-task model, performing both binary classification AND multi-class classification simultaneously, distinguishing
        # normal tissue from breast cancer tissue, as well as separating different types of breast cancer tissue
        some_input = Input(shape=instance_size)
        base_model = InceptionV3(include_top=False, weights="imagenet", pooling=None, input_tensor=some_input)
        base_model.trainable = False
        x = base_model.output
        x = Flatten()(x)

        # first output branch
        y1 = Dense(64)(x)
        y1 = BatchNormalization()(y1)
        y1 = Dropout(0.5)(y1)
        y1 = Activation('relu')(y1)
        y1 = Dense(num_stations[0], activation='softmax', name="cl1")(y1)

        # second output branch
        y2 = Dense(64)(x)
        y2 = BatchNormalization()(y2)
        y2 = Dropout(0.5)(y2)
        y2 = Activation('relu')(y2)
        y2 = Dense(num_stations[1], activation='softmax', name="cl2")(y2)

        model = Model(inputs=base_model.input,
                      outputs=[y1, y2])  # example of multi-task network through the functional API

    elif model_arch == "mobilenet":
        # MobileNetV2
        # For now copied recipe from above and replaced model with MobileNetV2
        some_input = Input(shape=instance_size)
        base_model = MobileNetV2(
            # input_shape=None, alpha=1.0,
            include_top=False, weights="imagenet",
            input_tensor=some_input, pooling=None,
            # classes=1000, classifier_activation="softmax",
        )
        # base_model = InceptionV3(include_top=False, weights="imagenet", pooling=None, input_tensor=some_input)
        base_model.trainable = False  # Freeze model
        x = base_model.output
        x = Flatten()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Activation('relu')(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)  # example of creation of TF-Keras model using the functional API

    elif model_arch == "mobilenet_with_preprocessing":

        if num_stations > 2:
            prediction_layer = Dense(num_stations, activation='softmax')  # multiclass
        else:
            prediction_layer = Dense(1, activation='sigmoid')  # binary

        # Create the base model from the pre-trained model MobileNet V2
        base_model = tf.keras.applications.MobileNetV2(input_shape=instance_size,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = Model(inputs=base_model.input, outputs=outputs)


    elif model_arch == "cvc_net":

        base_model = CVCNet(include_top=False, input_shape=instance_size, classes=num_stations)
        # base_model.trainable = False
        x = base_model.output
        x = Conv2D(num_stations, (1, 1))(x)
        x = PReLU()(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation("softmax", name="predictions")(x)
        model = Model(inputs=base_model.input,
                      outputs=x)

    elif model_arch == "vgg16":
        base_model = VGG16(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)
        base_model.trainable = False
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)


    elif model_arch == "vgg16_v2":

        base_model = VGG16(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)
        base_model.trainable = False

        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)

    elif model_arch == 'cnn-lstm':
        # Create the base model from the pre-trained model VGG16
        base_model = InceptionV3(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)

        # Freeze two convolution blocks
        for layer in base_model.layers[:-2]:
            layer.trainable = False

        # Make sure the correct layers are frozen
        #for i, layer in enumerate(base_model.layers):
            #print(i, layer.name, layer.trainable)

        # Create the input layer for the sequence of images
        sequence_input = Input(shape=(None, *instance_size))  # (B, T, H, W, C)

        # Create a Lambda layer to apply preprocessing to each image in the sequence
        preprocess = Lambda(lambda z: tf.keras.applications.inception_v3.preprocess_input(z))(sequence_input)

        # Apply the CNN base model to each image in the sequence
        x = TimeDistributed(base_model)(preprocess)  # (B, T, H', W', C')

        # Apply Global Average Pooling to each frame in the sequence
        x = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)  # (B, T, C')

        # Create an LSTM layer
        x = LSTM(32, return_sequences=False, stateful=stateful)(x)  # (B, lstm_output_dim)

        # Create a dense layer
        x = Dense(128, activation='relu')(x)  # (B, dense_output_dim)

        # Create a dropout layer
        x = Dropout(0.5)(x)  # (B, dense_output_dim)

        # Create the output layer for classification
        output = Dense(num_stations, activation='softmax')(x)  # (B, num_classes)

        # Create the combined model
        model = Model(inputs=sequence_input, outputs=output)


    else:
        print("please choose supported models: {basic, inception, resnet, inception_multi-task, mobilenet,"
              "cvc_net, vgg16, cnn-lstm}")
        exit()

    return model
