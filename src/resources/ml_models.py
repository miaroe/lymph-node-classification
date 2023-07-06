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

def get_arch(model_name, instance_size, num_classes):

    # basic
    if model_name == "basic":
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
        model.add(Dense(num_classes, activation='softmax'))

    elif model_name == "inception":
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
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)  # example of creation of TF-Keras model using the functional API

    elif model_name == "resnet":
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
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)

    elif model_name == "inception_multi-task":
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
        y1 = Dense(num_classes[0], activation='softmax', name="cl1")(y1)

        # second output branch
        y2 = Dense(64)(x)
        y2 = BatchNormalization()(y2)
        y2 = Dropout(0.5)(y2)
        y2 = Activation('relu')(y2)
        y2 = Dense(num_classes[1], activation='softmax', name="cl2")(y2)

        model = Model(inputs=base_model.input, outputs=[y1, y2])  # example of multi-task network through the functional API

    elif model_name == "mobilenet":
        # MobileNetV2
        # For now copied recipe from above and replaced model with MobileNetV2
        some_input = Input(shape=instance_size)
        base_model = MobileNetV2(
            #input_shape=None, alpha=1.0,
            include_top=False, weights="imagenet",
            input_tensor=some_input, pooling=None,
            #classes=1000, classifier_activation="softmax",
        )
        #base_model = InceptionV3(include_top=False, weights="imagenet", pooling=None, input_tensor=some_input)
        base_model.trainable = False    # Freeze model
        x = base_model.output
        x = Flatten()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Activation('relu')(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)  # example of creation of TF-Keras model using the functional API

    elif model_name == "mobilenet_with_preprocessing":
        # adapted from https://www.tensorflow.org/tutorials/images/transfer_learning
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2)
        ])
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # Create the base model from the pre-trained model MobileNet V2
        base_model = tf.keras.applications.MobileNetV2(input_shape=instance_size,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False

        inputs = tf.keras.Input(shape=instance_size)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)


    elif model_name == "cvc_net":

        base_model = CVCNet(include_top=False, input_shape=instance_size, classes=num_classes)
        #base_model.trainable = False
        x = base_model.output
        x = Conv2D(num_classes, (1, 1))(x)
        x = PReLU()(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation("softmax", name="predictions")(x)
        model = Model(inputs=base_model.input,
                      outputs=x)

    elif model_name == "vgg16":
        base_model = VGG16(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)
        base_model.trainable = False
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)

    elif model_name == "vgg16_v2":
        base_model = VGG16(include_top=False, weights="imagenet", pooling=None, input_shape=instance_size)
        base_model.trainable = False
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input,
                      outputs=x)



    else:
        print("please choose supported models: {basic, inception, resnet, inception_multi-task, mobilenet,"
              "cvc_net, vgg16}")
        exit()

    return model

