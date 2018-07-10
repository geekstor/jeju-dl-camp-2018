import tensorflow as tf
from tensorflow import layers

from configuration import ConfigurationManager


# General Neural Network (Convolutional or MLP).


class GeneralNetwork:
    required_params = ["CONVOLUTIONAL_LAYER_SPEC",
                       "DENSE_LAYER_SPEC",
                       "STATE_DTYPE"]

    def __init__(self, config_parser: ConfigurationManager):

        from tensorflow.contrib import framework
        print(framework.get_name_scope())

        self.cfg = config_parser.parse_and_return_dictionary(
            "NETWORK", GeneralNetwork.required_params)

        if "STATE_DIMENSIONS" in self.cfg:
            obs_shape = [int(i) for i in self.cfg["STATE_DIMENSIONS"]]
        else:
            obs_shape = config_parser.parsed_json["DEFAULT_OBS_DIMS"]

        # Input
        self.x = tf.placeholder(name="state",
                                dtype=tf.float32,
                                shape=(None, *obs_shape))

        # Convolutional Layers
        self.conv_outputs = []
        for CONV_LAYER_SPEC in self.cfg["CONVOLUTIONAL_LAYERS_SPEC"]:
            self.conv_outputs.append(
                layers.conv2d(
                    name="conv_layer_" + str(len(self.conv_outputs) + 1),
                    inputs=self.x if len(self.conv_outputs) == 0 else
                    self.conv_outputs[-1],
                    filters=CONV_LAYER_SPEC["filters"],
                    kernel_size=CONV_LAYER_SPEC["kernel_size"],
                    strides=CONV_LAYER_SPEC["strides"],
                    activation=tf.nn.relu
                )
            )
            print(self.conv_outputs[-1].shape.dims)

        if len(self.cfg["CONVOLUTIONAL_LAYERS_SPEC"]) > 0:
            # Flatten
            self.flattened_conv_output = tf.layers.flatten(
                name="conv_output_flattener", inputs=self.conv_outputs[-1]
            )

            last_out = self.flattened_conv_output
        else:
            last_out = self.x


        # Hidden Layer
        self.dense_outputs = []
        for DENSE_LAYER_SPEC in self.cfg["DENSE_LAYERS_SPEC"]:
            self.dense_outputs.append(
                layers.dense(
                    name="fc_layer_" + str(len(self.dense_outputs) + 1),
                    inputs=last_out if
                    len(self.dense_outputs) == 0 else
                    self.dense_outputs[-1], 
                    units=DENSE_LAYER_SPEC, activation=tf.nn.relu
                )
            )

        self.last_op = self.dense_outputs[-1]
