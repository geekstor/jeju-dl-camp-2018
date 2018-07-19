import random
import tensorflow as tf

import agent
from configuration import ConfigurationManager
from function_approximator.head import FixedAtomsDistributionalHead


class QuantileRegressionAgent(agent.DistributionalAgent):
    required_params = ["UPDATE_FREQUENCY", "DISCOUNT_FACTOR", "NB_ATOMS"]

    head = FixedAtomsDistributionalHead

    def __init__(self, cfg_parser: ConfigurationManager):
        super().__init__(cfg_parser, QuantileRegressionAgent.head)
        self.cfg = cfg_parser.parse_and_return_dictionary(
            "AGENT", QuantileRegressionAgent.required_params)

        self.build_loss_op()

        self.prepare(self.loss)

    def build_loss_op(self):
        batch_dim_range = tf.range(tf.shape(self.target_network_base.x)[0], dtype=tf.int32)

        self.q_vals_target = tf.reduce_mean(self.target_network.y, axis=-1)

        self.greedy_action_target = tf.cast(tf.argmax(self.q_vals_target, axis=-1),
                                            dtype=tf.int32)

        self.flat_indices_for_argmax_action_target_net = tf.stack([batch_dim_range,
                                                                   tf.cast(self.greedy_action_target, dtype=tf.int32)], axis=1)
        self.dist_of_greedy_actions_target_net = \
            tf.reshape(tf.gather_nd(self.target_network.y,
                                    self.flat_indices_for_argmax_action_target_net),
                       [-1, self.train_network.cfg["NB_ATOMS"]])

        self.expected_quantiles = self.reward_placeholder[:, tf.newaxis] + self.cfg["DISCOUNT_FACTOR"] * \
                                  tf.cast(self.terminal_placeholder[:, tf.newaxis], dtype=tf.float32) * \
                                  self.dist_of_greedy_actions_target_net

        # OPS. for computing dist. from current train net
        self.flat_indices_chosen_actions = tf.stack([batch_dim_range,
                                                     self.action_placeholder], axis=1)
        self.dist_of_chosen_actions = tf.reshape(tf.gather_nd(self.train_network.y,
                                                              self.flat_indices_chosen_actions),
                                                 [-1, self.train_network.cfg["NB_ATOMS"]])

        tau_hat = tf.linspace(0.0, 1.0 - 1. / self.train_network.cfg["NB_ATOMS"],
                              self.train_network.cfg["NB_ATOMS"]) + 0.5 / self.train_network.cfg["NB_ATOMS"]
        self.quantile_midpoints = tf.tile(tf.reshape(tau_hat, [1, -1]), (tf.shape(self.train_network_base.x)[0], 1))

        # Compute loss.
        u = tf.stop_gradient(self.expected_quantiles) - self.dist_of_chosen_actions
        k = 1
        huber_loss = 0.5 * tf.square(tf.clip_by_value(tf.abs(u), 0.0, k))
        huber_loss += k * (tf.abs(u) - tf.clip_by_value(tf.abs(u), 0.0, k))
        quantile_loss = tf.abs(self.quantile_midpoints - tf.cast((u < 0), tf.float32)) * \
                        huber_loss
        self.loss = tf.reduce_sum(quantile_loss) / self.train_network.cfg["NB_ATOMS"]

        self.q_vals_online = tf.reduce_mean(self.train_network.y, axis=-1)

        self.train_network.argmax_action = tf.cast(tf.argmax(self.q_vals_online, axis=-1),
                                            dtype=tf.int32)

    def distribution(self, state):
        return self.sess.run(fetches=[self.train_network.y],
                             feed_dict={self.train_network_base.x: state})

    def greedy_action(self, state):
        return self.sess.run(fetches=self.train_network.argmax_action,
                             feed_dict={self.train_network_base.x: state})

    def act(self, x):
        if random.random() < 1.0 - (min(10000, self.num_updates) / 10000) * (1 - 0.1):
            return random.randint(0, self.cfg_parser["NUM_ACTIONS"] - 1)
        else:
            return self.greedy_action([x])[0]
