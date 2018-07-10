import tensorflow as tf
from tensorflow import layers

from configuration import ConfigurationManager
from . import network


# Head for Reinforcement Learning
class Head:
    required_params = []

    def __init__(self, config_parser: ConfigurationManager):
        self.cfg = config_parser.parse_and_return_dictionary(
            "HEAD", Head.required_params)

        if "ACTION_SPECIFICATIONS" in self.cfg:
            self.actions = self.cfg["ACTION_SPECIFICATIONS"]
        else:
            self.actions = list(range(config_parser.parsed_json["DEFAULT_NUM_ACTIONS"]))

    def act_to_send(self, action):
        # TODO: Update Agent to use this function.
        # TODO: Important for games where action space is modified!

        return self.actions[action]


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
            units=len(self.actions) * self.cfg["NB_ATOMS"],
            activation=None
        )

        # Unflatten
        self.y = tf.reshape(self.flattened_dist, [-1, len(self.actions),
            self.cfg["NB_ATOMS"]], name="per_action_dist")


# For Categorical DQN.
class SoftmaxFixedAtomsDistributionalHead(FixedAtomsDistributionalHead):
    def __init__(self, config_parser: ConfigurationManager,
                 net: network.GeneralNetwork):
        super().__init__(config_parser, net)

        self.cfg = config_parser.parse_and_return_dictionary(
            "HEAD", Head.required_params)

        self.y = tf.nn.softmax(self.y, name="state_action_value_dist",
                              axis=-1)

class IQNHead(Head):
    # N comes from agent parameter.
    def __init__(self, config_parser: ConfigurationManager,
                 net: network.GeneralNetwork, N):
        super().__init__(config_parser)

        self.psi = net.last_op

        self.tau = tf.random_uniform(shape=[N,])

