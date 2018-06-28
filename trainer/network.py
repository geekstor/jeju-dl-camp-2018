import tensorflow as tf

from configuration import params_cartpole as params


class DistributionalAgentNet:
    def __init__(self):
        num_actions = len(params.GLOBAL_MANAGER.actions)
        # Input
        self.x = tf.placeholder(name="state",
                                dtype=params.STATE_DTYPE,
                                shape=(None, *params.STATE_DIMENSIONS))

        if hasattr(params, "SCALED_FLOAT") and not params.SCALED_FLOAT:
            self.normalized_x = tf.cast(self.x, dtype=tf.float32) / 255.0
        else:
            self.normalized_x = self.x

        with tf.variable_scope("common"):
            # Convolutional Layers
            self.conv_outputs = []
            for CONV_LAYER_SPEC in params.CONVOLUTIONAL_LAYERS_SPEC:
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

        if len(params.CONVOLUTIONAL_LAYERS_SPEC) > 0:
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
        for DENSE_LAYER_SPEC in params.DENSE_LAYERS_SPEC:
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
            num_outputs=num_actions * params.NB_ATOMS,
            activation_fn=None
        )

        # Unflatten
        self.y = tf.reshape(
            self.flattened_dist, [-1, num_actions,
                                  params.NB_ATOMS],
            name="per_action_dist"
        )
