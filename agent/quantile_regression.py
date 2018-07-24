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
        self.expected_quantiles = self.bellman_op(self.dist_of_target_max_q)

        tau_hat = tf.range(0.0, self.train_network.cfg["NB_ATOMS"] + 1)
        tau_hat = tau_hat * 1. / self.train_network.cfg["NB_ATOMS"]
        self.tau_hat = (tau_hat[:-1] + tau_hat[1:]) / 2.

        # Compute loss.
        u = self.expected_quantiles[:, tf.newaxis, :] - \
            self.dist_of_chosen_actions[:, :, tf.newaxis]

        from util.util import asymmetric_huber_loss
        self.loss = tf.reduce_mean(asymmetric_huber_loss(u, self.cfg["KAPPA"],
                                                         self.tau_hat))

        return self.loss

    def distribution(self, state):
        return self.sess.run(fetches=[self.train_network.q_dist],
                             feed_dict={self.train_network_base.x: state})

    def viz_dist(self, x):
        import numpy as np
        inp = np.array(x)
        feed_dict = {self.train_network_base.x: [inp]}

        tau_hat = np.linspace(0.0, 1.0, self.train_network.cfg["NB_ATOMS"])

        h = self.sess.run(fetches=self.train_network.q_dist[0],
                        feed_dict=feed_dict)

        from matplotlib import pyplot as plt
        # plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=2)
        # # plt.subplot(len(self.train_network.actions), 2, [1, 3])
        # from scipy.misc import imresize
        # plt.imshow(inp[:, :, 0], interpolation="nearest", cmap="gray")

        for i in range(h.shape[0]):
            plt.subplot2grid((2, 1), (i, 0), colspan=1, rowspan=1)
            plt.stem(tau_hat, h[i], markerfmt=" ")

        plt.pause(0.1)
        plt.gcf().clear()

        data = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

        return data
