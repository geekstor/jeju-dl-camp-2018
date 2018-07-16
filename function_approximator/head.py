import tensorflow as tf
from tensorflow import layers

from configuration import ConfigurationManager
from . import network


# Head for Reinforcement Learning
class Head:
    required_params = []

    def __init__(self, config_parser: ConfigurationManager):
        self.cfg = config_parser.parse_and_return_dictionary(
            "ENVIRONMENT", Head.required_params)

        if "ACTION_SPECIFICATIONS" in self.cfg:
            self.num_actions = len(self.cfg["ACTION_SPECIFICATIONS"])
        else:
            self.num_actions = config_parser.parsed_json["DEFAULT_NUM_ACTIONS"]

# For Quantile Regression DQN.
class FixedAtomsDistributionalHead(Head):
    required_params = ["NB_ATOMS"]

    def __init__(self, config_parser: ConfigurationManager,
                 net: network.GeneralNetwork):
        super().__init__(config_parser)

        self.cfg = config_parser.parse_and_return_dictionary(
            "HEAD", FixedAtomsDistributionalHead.required_params)

        # State-Action-Value Distributions (as a flattened vector)
        self.flattened_dist = layers.dense(
            name="flattened_dists",
            inputs=net.last_op,
            units=self.num_actions * self.cfg["NB_ATOMS"],
            activation=None
        )

        # Unflatten
        self.y = tf.reshape(self.flattened_dist, [-1, self.num_actions,
            self.cfg["NB_ATOMS"]], name="per_action_dist")


# For Categorical DQN.
class SoftmaxFixedAtomsDistributionalHead(FixedAtomsDistributionalHead):
    required_params = ["NB_ATOMS"]

    def __init__(self, config_parser: ConfigurationManager,
                 net: network.GeneralNetwork):
        super().__init__(config_parser, net)

        self.cfg = config_parser.parse_and_return_dictionary(
            "HEAD", SoftmaxFixedAtomsDistributionalHead.required_params)

        self.y = tf.nn.softmax(self.y, name="state_action_value_dist",
                              axis=-1)


class IQNHead(Head):
    required_params = ["EMBEDDING_SIZE"]

    def __init__(self, config_parser: ConfigurationManager,
                 net: network.GeneralNetwork):
        super().__init__(config_parser)

        self.cfg = config_parser.parse_and_return_dictionary(
            "HEAD", IQNHead.required_params)

        self.psi = net.last_op

        self.num_samples = tf.placeholder(dtype=tf.int32, shape=[],
                                          name="num_samples")

        self.tau = tf.random_uniform(shape=[tf.shape(self.psi)[0], self.num_samples],
                                minval=0, maxval=1,
                                dtype=tf.float32)
        import math as m
        pi = tf.constant(m.pi)

        self.phi = tf.layers.dense(inputs=tf.cos(tf.einsum('bn,j->bnj', self.tau,
        tf.range(self.cfg["EMBEDDING_SIZE"], dtype=tf.float32)) * pi), units=64,
        activation=tf.nn.relu)

        mul = tf.einsum('bnj,bj->bnj', self.phi, self.psi)

        self.q_dist = tf.transpose(
            tf.layers.dense(inputs=mul, units=self.num_actions, activation=None),
            perm=[0, 2, 1]
        )

        self.q = tf.reduce_mean(self.q_dist, axis=-1)

        self.greedy_action = tf.cast(tf.squeeze(
            tf.argmax(self.q, axis=-1)
        ), dtype=tf.int32)
