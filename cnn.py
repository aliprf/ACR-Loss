from config import DatasetName, DatasetType, W300Conf, InputDataSize, LearningConfig
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from keras.regularizers import l2, l1
from keras.models import Model
from keras.applications import mobilenet_v2, mobilenet, densenet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input, \
    GlobalMaxPool2D
import efficientnet.tfkeras as efn


class CNNModel:
    def get_model(self, arch, output_len):

        if arch == 'mobileNetV2':
            model = self.create_MobileNet(inp_shape=[224, 224, 3], output_len=output_len)

        elif arch == 'efNb0':
            model = self.create_efficientNet_b0(inp_shape=[224, 224, 3], input_tensor=None, output_len=output_len)

        elif arch == 'efNb3':
            model = self.create_efficientNet_b3(inp_shape=[224, 224, 3], input_tensor=None, output_len=output_len)

        return

    def create_MobileNet(self, inp_shape, output_len):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   pooling=None)
        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d').output  # 1280
        x = Dropout(0.1)(x)
        out_landmarks = Dense(output_len, activation=keras.activations.linear, kernel_initializer=initializer,
                              use_bias=True, name='O_L')(x)
        inp = mobilenet_model.input

        revised_model = Model(inp, [out_landmarks])

        revised_model.summary()
        model_json = revised_model.to_json()

        with open("mobileNet_v2.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def create_efficientNet_b0(self, inp_shape, input_tensor, output_len):
        initializer = tf.keras.initializers.he_uniform()

        eff_net = efn.EfficientNetB0(include_top=True,
                                     weights=None,
                                     input_tensor=input_tensor,
                                     input_shape=inp_shape,
                                     pooling=None)
        eff_net.layers.pop()
        inp = eff_net.input

        x = eff_net.get_layer('top_activation').output
        x = GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        output = Dense(output_len, activation='linear', use_bias=True, name='out',
                       kernel_initializer=initializer)(x)

        eff_net = Model(inp, output)
        eff_net.summary()

        model_json = eff_net.to_json()
        with open("eff_net_b0.json", "w") as json_file:
            json_file.write(model_json)

        return eff_net

    def create_efficientNet_b3(self, inp_shape, input_tensor, output_len):
        initializer = tf.keras.initializers.he_uniform()

        eff_net = efn.EfficientNetB3(include_top=True,
                                     weights=None,
                                     input_tensor=input_tensor,
                                     input_shape=inp_shape,
                                     pooling=None)
        eff_net.layers.pop()
        inp = eff_net.input

        x = eff_net.get_layer('top_activation').output
        x = GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(rate=0.5)(x)

        output = Dense(output_len, activation='linear', use_bias=True, name='out',
                       kernel_initializer=initializer)(x)

        eff_net = Model(inp, output)
        eff_net.summary()

        model_json = eff_net.to_json()
        with open("eff_net_b3.json", "w") as json_file:
            json_file.write(model_json)

        return eff_net
