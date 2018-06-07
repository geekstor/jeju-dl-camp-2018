import numpy as np
import tensorflow as tf


class C51Agent():
    def __init__(self):
        self.x = tf.placeholder(name="state", dtype=tf.float32, shape=(None, 84, 84, 4))
        self.r = tf.placeholder(name="reward", dtype=tf.float32, shape=(None, 1))

        self.conv_l1_out = tf.layers.conv2d(name="conv_layer_1",
                                            inputs=self.x,
                                            filters=32, kernel_size=8,
                                            strides=4, activation=tf.nn.relu)
        self.conv_l2_out = tf.layers.conv2d(name="conv_layer_2",
                                            inputs=self.conv_l1_out,
                                            filters=64, kernel_size=4,
                                            strides=2, activation=tf.nn.relu)
        self.conv_l3_out = tf.layers.conv2d(name="conv_layer_3",
                                            inputs=self.conv_l2_out,
                                            filters=64, kernel_size=3,
                                            strides=1, activation=tf.nn.relu)
        self.flattened_conv_output = tf.layers.flatten(name="conv_output_flattenner",
                                                       inputs=self.conv_l3_out)
        self.hidden_l_out = tf.layers.dense(name="dense_layer",
                                            inputs=self.flattened_conv_output,
                                            units=512)
        self.flattened_q_dist = tf.layers.dense(name="flattened_action_value_dist_logits",
                                                inputs=self.hidden_l_out,
                                                units=18 * 51)
        self.q_dist_logits = tf.reshape(self.flattened_q_dist,
                                        [-1, 18, 51],
                                        name="reshape_q_dist_logits")
        self.q_dist = tf.nn.softmax(self.q_dist_logits,
                                    name="action_value_dist")

    def act(self, x):
        # REQUIRES: x: shape{Batch Size, Width, Height, Hist. Len.}
        # MODIFIES: None
        # EFFECTS: return action
        pass

    def update(self, x, a, r, x_prime, t): # By Convention: SARSA
        pass
        # See: Algorithm 1 (Categorical Algorithm)


def test_agent_forward():
    c = C51Agent()
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        return sess.run(c.q_dist, feed_dict={c.x: np.ones([1, 4, 84, 84])})


test_agent_forward()

# TODO: Implement act() and update() in C51Agent(). Need to ensure two copies
# TODO: are made (train and target)

# TODO: In another 2 files, implement utility functions (pre-process,
# TODO: frame history buffer-handling) and experience replay functions.

