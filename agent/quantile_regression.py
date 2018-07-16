import random

import matplotlib.pyplot as plt
import numpy as np
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

        self.build_networks()

        self.prepare()

    def build_networks(self):
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

        self.r = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.t = tf.placeholder(shape=[None, ], dtype=tf.uint8)

        self.expected_quantiles = self.r[:, tf.newaxis] + self.cfg["DISCOUNT_FACTOR"] * \
                                  tf.cast(self.t[:, tf.newaxis], dtype=tf.float32) * \
                                  self.dist_of_greedy_actions_target_net

        # OPS. for computing dist. from current train net
        self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, ])
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

        from optimizer.optimizer import get_optimizer
        self.train_step = get_optimizer(cfg_parser=self.cfg_parser, loss_op=self.loss,
                                  var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                              scope='train_net'))

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
            return random.randint(0, self.train_network.num_actions - 1)
        else:
            return self.greedy_action([x])[0]

    def viz(self, x, rgb_x):
        plt.switch_backend("Agg")
        # Plot
        h = np.squeeze(self.sess.run(fetches=self.train_network.y,
                                     feed_dict={self.train_network_base.x: x}))

        plt.subplot2grid((h.shape[0], h.shape[0]), (0, 0), colspan=1, rowspan=h.shape[0])
        # plt.subplot(len(self.train_network.actions), 2, [1, 3])
        from scipy.misc import imresize
        plt.imshow(imresize(rgb_x, [rgb_x.shape[0] * 10, rgb_x.shape[1] * 10]),
                   aspect="auto", interpolation="nearest")

        l, s = np.linspace(0,
                           1,
                           self.train_network.cfg["NB_ATOMS"],
                           retstep=True)

        for i in range(h.shape[0]):
            plt.subplot2grid((h.shape[0], h.shape[0]), (i, 1), colspan=1, rowspan=1)
            # plt.subplot(len(self.train_network.actions), 2, 2 * (i + 1))
            plt.bar(l - s / 2., height=h[i], width=s,
                    color="brown", edgecolor="red", linewidth=0.5, align="edge")

        #plt.pause(0.1)
        #plt.gcf().clear()

        data = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

        return data
