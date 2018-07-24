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

        self.greedy_action = None


class QNetworkHead(Head):
    def __init__(self, config_parser: ConfigurationManager):
        super().__init__(config_parser)
        self.q = None


class DistributionalHead(QNetworkHead):
    def __init__(self, config_parser: ConfigurationManager):
        super().__init__(config_parser)
        self.q_dist = None


# For Quantile Regression DQN.
class FixedAtomsDistributionalHead(DistributionalHead):
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
        self.q_dist = tf.reshape(self.flattened_dist, [-1, self.num_actions,
            self.cfg["NB_ATOMS"]], name="per_action_dist")

        self.q = tf.reduce_mean(self.q_dist, axis=-1)

        self.greedy_action = tf.cast(tf.squeeze(
            tf.argmax(self.q, axis=-1)
        ), dtype=tf.int32)


# For Categorical DQN.
class SoftmaxFixedAtomsDistributionalHead(FixedAtomsDistributionalHead):
    required_params = ["NB_ATOMS"]

    def __init__(self, config_parser: ConfigurationManager,
                 net: network.GeneralNetwork):
        super().__init__(config_parser, net)

        self.cfg = config_parser.parse_and_return_dictionary(
            "HEAD", SoftmaxFixedAtomsDistributionalHead.required_params)

        self.q_dist = tf.nn.softmax(self.q_dist, name="state_action_value_dist",
                              axis=-1)


class IQNHead(DistributionalHead):
    required_params = ["EMBEDDING_SIZE"]

    def __init__(self, config_parser: ConfigurationManager,
                 net: network.GeneralNetwork):
        super().__init__(config_parser)

        self.cfg = config_parser.parse_and_return_dictionary(
            "HEAD", IQNHead.required_params)

        self.psi = net.last_op

        self.num_samples = tf.placeholder(dtype=tf.int32, shape=[],
                                          name="num_samples")

        # Preprocessed tau (choose number of samples and pass through beta as necessary)
        from action_policy.distorted_expectation import distorted_expectation, get_uniform_dist
        self.uniform_tau = get_uniform_dist(psi=self.psi, N_placeholder=self.num_samples)

        self.distorted_tau = distorted_expectation(config_parser,
                                         psi=self.psi, N_placeholder=self.num_samples)
        import math as m
        pi = tf.constant(m.pi)

        cos_embed = tf.layers.Dense(units=self.cfg["EMBEDDING_SIZE"],
                                    activation=tf.nn.relu, name="cosine_embedding")

        self.distorted_tau_phi = cos_embed(tf.cos(tf.einsum('bn,j->bnj',
                                                  self.distorted_tau,
                                                  tf.range(self.cfg["EMBEDDING_SIZE"],
                                                           dtype=tf.float32)) * pi))

        mul_distorted = tf.einsum('bnj,bj->bnj', self.distorted_tau_phi, self.psi)

        ###

        self.uniform_tau_phi = cos_embed(tf.cos(tf.einsum('bn,j->bnj',
                                                self.uniform_tau,
                                         tf.range(self.cfg["EMBEDDING_SIZE"],
                                                  dtype=tf.float32)) * pi))

        mul_uniform = tf.einsum('bnj,bj->bnj', self.uniform_tau_phi, self.psi)

        ###

        q_dist_layer = tf.layers.Dense(units=self.num_actions,
                        activation=None, name="q_dist")

        self.q_dist = tf.transpose(
            q_dist_layer(mul_uniform),
            perm=[0, 2, 1]
        )

        self.q_undistorted = tf.reduce_mean(self.q_dist, axis=-1)

        self.q_dist_distorted = tf.transpose(
            q_dist_layer(mul_distorted),
            perm=[0, 2, 1]
        )

        self.q = tf.reduce_mean(self.q_dist_distorted, axis=-1)

        self.greedy_action = tf.cast(tf.squeeze(
            tf.argmax(self.q, axis=-1)
        ), dtype=tf.int32)
