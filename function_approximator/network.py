import tensorflow as tf

from configuration import ConfigurationManager


# General Neural Network (Convolutional or MLP) for RL with support for distributions
# using NB_ATOMS as a parameter.


class GeneralNetwork:
    required_params = ["CONVOLUTIONAL_LAYER_SPEC",
                       "DENSE_LAYER_SPEC", "NB_ATOMS",
                       "STATE_DTYPE"]

    def __init__(self, config_parser: ConfigurationManager,
                 default_num_actions, default_obs_shape,
                 more_networks_needed=False):
        if more_networks_needed:
            print("Assuming use of multiple networks.")
        else:
            print("Network section will be removed. "
                  "Assuming all networks needed built.")

        self.cfg = config_parser.parse_and_return_dictionary(
            "NETWORK", GeneralNetwork.required_params,
            keep_section=more_networks_needed)

        if "ACTION_SPECIFICATIONS" in self.cfg:
            self.actions = self.cfg["ACTION_SPECIFICATIONS"]
        else:
            self.actions = list(range(default_num_actions))

        if "STATE_DIMENSIONS" in self.cfg:
            obs_shape = self.cfg["STATE_DIMENSIONS"]
        else:
            obs_shape = default_obs_shape

        # Input
        self.x = tf.placeholder(name="state",
                                dtype=self.cfg["STATE_DTYPE"],
                                shape=(None, *obs_shape))

        # For Atari Environments, wrappers for DeepMind style experiments
        # take a parameter SCALED_FLOAT to scale inputs to [0, 1]. This is
        # not recommended to be used with large replay memories. Instead, a cast
        # to float and normalization prior to forward is desired.
        if "SCALED_FLOAT" in self.cfg and not self.cfg["SCALED_FLOAT"]:
            self.normalized_x = tf.cast(self.x, dtype=tf.float32) / 255.0
        else:
            self.normalized_x = self.x

        with tf.variable_scope("common"):
            # Convolutional Layers
            self.conv_outputs = []
            for CONV_LAYER_SPEC in self.cfg["CONVOLUTIONAL_LAYERS_SPEC"]:
                self.conv_outputs.append(
                    tf.layers.conv2d(
                        name="conv_layer_" + str(len(self.conv_outputs) + 1),
                        inputs=self.normalized_x if len(self.conv_outputs) == 0 else
                        self.conv_outputs[-1],
                        filters=CONV_LAYER_SPEC["filters"],
                        kernel_size=CONV_LAYER_SPEC["kernel_size"],
                        strides=CONV_LAYER_SPEC["strides"],
                        activation=tf.nn.relu
                    )
                )

        if len(self.cfg["CONVOLUTIONAL_LAYERS_SPEC"]) > 0:
            # Flatten
            self.flattened_conv_output = tf.layers.flatten(
                name="conv_output_flattener", inputs=self.conv_outputs[-1]
            )

            last_out = self.flattened_conv_output
        else:
            last_out = self.normalized_x

        from tensorflow.contrib.layers import fully_connected
        # Hidden Layer
        self.dense_outputs = []
        for DENSE_LAYER_SPEC in self.cfg["DENSE_LAYERS_SPEC"]:
            self.dense_outputs.append(
                fully_connected(
                    #name="fc_layer_" + str(len(self.dense_outputs) + 1),
                    inputs=last_out if
                    len(self.dense_outputs) == 0 else
                    self.dense_outputs[-1],
                    num_outputs=DENSE_LAYER_SPEC, activation_fn=tf.nn.relu
                )
            )

        # State-Action-Value Distributions (as a flattened vector)
        self.flattened_dist = fully_connected(
            #name="flattened_dists",
            inputs=self.dense_outputs[-1],
            num_outputs=len(self.actions) * self.cfg["NB_ATOMS"],
            activation_fn=None
        )

        # Unflatten
        self.y = tf.reshape(
            self.flattened_dist, [-1, len(self.actions),
                                  self.cfg["NB_ATOMS"]],
            name="per_action_dist"
        )

    def act_to_send(self, action):
        # TODO: Update Agent to use this function.
        # TODO: Important for games where action space is modified!
        return self.actions[action]
