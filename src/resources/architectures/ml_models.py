import tensorflow as tf
from mlmia.architectures import CVCNet
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape, LSTM, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import PReLU
from keras import regularizers
from src.resources.architectures.timingnet import TimingNet
from src.resources.architectures.ResNet18 import ResNet18


# see here for already built-in pretrained architectures:
# https://keras.io/api/applications/

def get_arch(model_arch, instance_size, num_stations, stateful=False):
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

    elif model_arch == "inception": # scale to [-1, 1]
        # InceptionV3 (typical example arch) - personal preference for CNN classification (however, quite expensive and might be overkill in a lot of scenarios)
        base_model = InceptionV3(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)
        #base_model.trainable = False

        #for layer in base_model.layers[249:]:
        #    if not isinstance(layer, BatchNormalization):
        #        layer.trainable = True

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        x = base_model.output
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)

    elif model_arch == "resnet": # scale [0, 255]
        # ResNet-50, another very popular arch, can be done similarly as for InceptionV3 above

        base_model = ResNet50(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)
        #base_model.trainable = False

        #for layer in base_model.layers[143:]:
        #    if not isinstance(layer, BatchNormalization):
        #        layer.trainable = True

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        x = base_model.output
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)

    elif model_arch == "mobilenetV2": # scale to [-1, 1]
        base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=instance_size, pooling=None)

        for layer in base_model.layers[:-11]:
            layer.trainable = False

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        x = base_model.output
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)

    elif model_arch == "mobileNetV3Small":

        if num_stations > 2:
            prediction_layer = Dense(num_stations, activation='softmax')  # multiclass
        else:
            prediction_layer = Dense(1, activation='sigmoid')  # binary

        # Create the base model from the pre-trained model MobileNet V2
        #base_model = tf.keras.applications.MobileNetV2(input_shape=instance_size,
        #                                               include_top=False,
        #                                               weights='imagenet')

        base_model = tf.keras.applications.MobileNetV3Small(input_shape=instance_size, include_top=False,
                                                            weights='imagenet', include_preprocessing=True,
                                                            minimalistic=True, dropout_rate=0.3)

        for layer in base_model.layers[:-6]:
            layer.trainable = False

        #base_model.trainable = False

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)


        x = base_model.output
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = prediction_layer(x)
        model = Model(inputs=base_model.input, outputs=outputs)


    elif model_arch == "cvc_net": #[0, 1]

        base_model = CVCNet(include_top=False, input_shape=instance_size, classes=num_stations, dropout_rate=0.4)

        x = base_model.output
        x = Conv2D(num_stations, (1, 1))(x)
        x = PReLU()(x)
        x = GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = Activation("softmax", name="predictions")(x)
        model = Model(inputs=base_model.input,
                      outputs=x)

    elif model_arch == "vgg16": # scale [0, 255], 224x224
        base_model = VGG16(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)
        #base_model.trainable = False
        #for layer in base_model.layers[:-4]:
        #    layer.trainable = False

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        #inputs = tf.keras.Input(shape=instance_size)
        #x = tf.keras.applications.vgg16.preprocess_input(inputs)
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)


    elif model_arch == "vgg16_v2":

        base_model = VGG16(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)
        #base_model.trainable = False

        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        #x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        #x = BatchNormalization()(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)

    elif model_arch== "efficientnet":

        base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet", input_shape=instance_size)
        base_model.trainable = False

        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        #for layer in base_model.layers[-16:]:
        #    if not isinstance(layer, BatchNormalization):
        #        layer.trainable = True

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)

    elif model_arch == "xception":

        base_model = tf.keras.applications.xception.Xception(include_top=False, weights="imagenet", input_shape=instance_size)
        base_model.trainable = False

        # We unfreeze the top 16 layers while leaving BatchNorm layers frozen
        for layer in base_model.layers[-16:]:
           if not isinstance(layer, BatchNormalization):
               layer.trainable = True

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)

    elif model_arch == "inception_resnet_v2":

        base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=instance_size)
        base_model.trainable = False

        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in base_model.layers[-6:]:
           if not isinstance(layer, BatchNormalization):
               layer.trainable = True

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_stations, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)


    elif model_arch == 'mobilenetV2-lstm':
        # Create the base model from the pre-trained model
        base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=instance_size, pooling=None)

        for layer in base_model.layers[:-11]:
            layer.trainable = False

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        # Create the input layer for the sequence of images
        sequence_input = Input(shape=(None, *instance_size))  # (B, T, H, W, C)

        # Apply the CNN base model to each image in the sequence
        x = TimeDistributed(base_model)(sequence_input)  # (B, T, H', W', C')

        # Apply Global Average Pooling to each frame in the sequence
        x = TimeDistributed(tf.keras.layers.GlobalMaxPooling2D())(x)  # (B, T, C')

        # Create an LSTM layer
        x = LSTM(64, return_sequences=True, stateful=stateful)(x)  # (B, T, lstm_output_dim)

        x = LSTM(64, return_sequences=False, stateful=stateful)(x)  # (B, lstm_output_dim)

        # Create a dense layer
        #x = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)  # (B, dense_output_dim)
        x = Dense(256, activation='relu')(x)  # (B, dense_output_dim)

        # Create a dropout layer
        x = Dropout(0.5)(x)  # (B, dense_output_dim)

        # Create the output layer for classification
        output = Dense(num_stations, activation='softmax')(x)  # (B, num_classes)

        # Create the combined model
        model = Model(inputs=sequence_input, outputs=output)

    elif model_arch == 'mobileNetV3Small-lstm':
        base_model = tf.keras.applications.MobileNetV3Small(input_shape=instance_size, include_top=False, weights='imagenet',
                                                            include_preprocessing=True, minimalistic=True, dropout_rate=0.3)

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        sequence_input = Input(shape=(None, *instance_size))  # (B, T, H, W, C)
        x = TimeDistributed(base_model)(sequence_input)  # (B, T, H', W', C')
        x = TimeDistributed(tf.keras.layers.GlobalMaxPooling2D())(x)  # (B, T, C')
        x = LSTM(64, return_sequences=True, stateful=stateful)(x)  # (B, T, lstm_output_dim)
        x = LSTM(64, return_sequences=False, stateful=stateful)(x)  # (B, lstm_output_dim)
        x = Dense(256, activation='relu')(x)  # (B, dense_output_dim)
        x = Dropout(0.5)(x)  # (B, dense_output_dim)
        output = Dense(num_stations, activation='softmax')(x)  # (B, num_classes)
        model = Model(inputs=sequence_input, outputs=output)

    elif model_arch == 'inception-lstm':
        base_model = InceptionV3(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        sequence_input = Input(shape=(None, *instance_size))  # (B, T, H, W, C)
        x = TimeDistributed(base_model)(sequence_input)  # (B, T, H', W', C')
        x = TimeDistributed(tf.keras.layers.GlobalMaxPooling2D())(x)  # (B, T, C')
        x = LSTM(128, return_sequences=True, stateful=stateful)(x)  # (B, T, lstm_output_dim)
        x = LSTM(128, return_sequences=False, stateful=stateful)(x)  # (B, lstm_output_dim)
        x = Dense(256, activation='relu')(x)  # (B, dense_output_dim)
        x = Dropout(0.5)(x)  # (B, dense_output_dim)
        output = Dense(num_stations, activation='softmax')(x)  # (B, num_classes)
        model = Model(inputs=sequence_input, outputs=output)

    elif model_arch == 'resnet-lstm':
        base_model = ResNet50(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)
        base_model.trainable = False

        for layer in base_model.layers[143:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True

        # Make sure the correct layers are frozen
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)

        sequence_input = Input(shape=(None, *instance_size))  # (B, T, H, W, C)
        x = TimeDistributed(base_model)(sequence_input)  # (B, T, H', W', C')
        x = TimeDistributed(tf.keras.layers.GlobalMaxPooling2D())(x)  # (B, T, C')
        x = LSTM(128, return_sequences=True, stateful=stateful)(x)  # (B, T, lstm_output_dim)
        x = LSTM(128, return_sequences=False, stateful=stateful)(x)  # (B, lstm_output_dim)
        x = Dense(256, activation='relu')(x)  # (B, dense_output_dim)
        x = Dropout(0.5)(x)  # (B, dense_output_dim)
        output = Dense(num_stations, activation='softmax')(x)  # (B, num_classes)
        model = Model(inputs=sequence_input, outputs=output)

    elif model_arch == 'timingnet': # scale to [0, 1]
        print('input shape: ', (None, *instance_size))

        model = TimingNet(input_shape=(None, *instance_size), num_stations=num_stations)

    elif model_arch == 'resnet18':
        model = ResNet18(input_shape=(None, *instance_size), num_stations=num_stations)

    elif model_arch == 'mutli-input_mobilenetV2-lstm':
        # Define the inputs
        image_input = Input(shape=(None, *instance_size), name='image_input')  # Sequence of images
        mask_input = Input(shape=(None, *instance_size), name='mask_input')  # Sequence of segmentation masks

        # Image processing pathway
        # Here, we're using a TimeDistributed wrapper to apply a model to each time step independently
        #image_model = tf.keras.applications.MobileNetV3Small(input_shape=instance_size, include_top=False, weights='imagenet',
        #                                                    include_preprocessing=True, minimalistic=True, dropout_rate=0.3)

        image_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=instance_size, pooling=None)

        for layer in image_model.layers[:-11]:
            layer.trainable = False

        image_features = TimeDistributed(image_model)(image_input)
        image_features = TimeDistributed(GlobalAveragePooling2D())(image_features)

        # Mask processing pathway
        # Simple CNN for extracting features from masks, adjust as necessary
        mask_features = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(mask_input)
        mask_features = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(mask_features)
        mask_features = TimeDistributed(GlobalAveragePooling2D())(mask_features)

        # Combine the features
        combined_features = Concatenate()([image_features, mask_features])

        # Create an LSTM layer
        lstm_1 = LSTM(64, return_sequences=True, stateful=stateful)(combined_features)  # (B, T, lstm_output_dim)

        lstm_2 = LSTM(64, return_sequences=False, stateful=stateful)(lstm_1)  # (B, lstm_output_dim)

        # Create a dense layer
        dense = Dense(256, activation='relu')(lstm_2)  # (B, dense_output_dim)

        # Create a dropout layer
        dropout = Dropout(0.5)(dense)  # (B, dense_output_dim)

        # Create the output layer for classification
        output = Dense(num_stations, activation='softmax')(dropout)  # (B, num_classes)

        # Create the model
        model = Model(inputs=[image_input, mask_input], outputs=output)


    else:
        print("please choose supported models: {basic, inception, resnet, mobilenetV2,"
              "mobileNetV3Small, cvc_net, vgg16, vgg16_v2, efficientnet, xception, inception_resnet_v2,"
              "mobilenetV2-lstm, mobileNetV3Small-lstm, inception-lstm, resnet-lstm, timingnet, resnet18, mutli-input_mobilenetV2-lstm}")
        exit()

    return model
