import agent
import tensorflow as tf
from configuration import ConfigurationManager
from function_approximator.head import IQNHead


class ImplicitQuantileAgent(agent.DistributionalAgent):
    required_params = ["UPDATE_FREQUENCY", "DISCOUNT_FACTOR",
                       "N", # Number of Samples of Tau to take when getting Q-value (online net)
                       "N_PRIME", # Number of Samples of Tau to take when calculating target
                       "K",
                       "KAPPA"] # Number of Samples of Tau to take when acting # TODO: Beta

    head = IQNHead

    def __init__(self, cfg_parser: ConfigurationManager):
        super().__init__(cfg_parser, ImplicitQuantileAgent.head)

        self.loss = self.build_loss_op()

        self.prepare(self.loss)

    def build_loss_op(self) -> tf.Operation:
        # OPS. for computing dist. from current train net
        expected_quantiles = self.bellman_op(self.dist_of_target_max_q)
        u = expected_quantiles[:, tf.newaxis, :] - \
            self.dist_of_chosen_actions[:, :, tf.newaxis]
        k = tf.constant(self.cfg["KAPPA"], name="kappa", dtype=tf.float32)
        from util.util import asymmetric_huber_loss
        loss = tf.reduce_mean(asymmetric_huber_loss(u, k,
                              self.train_network.uniform_tau[:, tf.newaxis, :]))

        diversity_bonus = tf.reduce_mean(
            tf.nn.moments(self.train_network.distorted_tau, axes=-1)[1]
        )

        p = tf.stack((self.train_network.distorted_tau,
                      (1. - self.train_network.distorted_tau)), axis=-1)

        extreme_loss = tf.reduce_mean(tf.reduce_sum(
            -p * tf.log(p) / tf.log(2.), axis=-1
        ))

        extreme_loss = tf.Print(extreme_loss, [p, extreme_loss, diversity_bonus], summarize=10)

        flat_indices_chosen_actions = tf.stack([self.batch_dim_range,
                                                self.action_placeholder],
                                               axis=1)
        distorted_dist_of_chosen_actions = tf.gather_nd(self.train_network.q_dist_distorted,
                                                   flat_indices_chosen_actions)

        u = expected_quantiles[:, tf.newaxis, :] - \
            distorted_dist_of_chosen_actions[:, :, tf.newaxis]
        k = tf.constant(self.cfg["KAPPA"], name="kappa", dtype=tf.float32)
        from util.util import asymmetric_huber_loss
        loss_2 = tf.reduce_mean(asymmetric_huber_loss(u, k,
                              distorted_dist_of_chosen_actions[:, tf.newaxis, :]))

        return loss + loss_2 #+ extreme_loss - diversity_bonus

    def learn(self, experiences):
        feed_dict = self.batch_experiences(experiences)
        feed_dict.update({
            self.train_network.num_samples: self.cfg["N"],
            self.target_network.num_samples: self.cfg["N_PRIME"]
        })
        loss, _, o_q, t_q  = self.sess.run(fetches=[self.loss, self.train_step,
                                         self.target_network.q_dist,
                                         self.train_network.q_dist], feed_dict=feed_dict)

    def viz_dist(self, x):
        import numpy as np
        inp = np.array(x)
        feed_dict = {self.train_network_base.x: [inp],
                     self.train_network.num_samples: self.cfg["K"]}
        tau, h = self.sess.run(fetches=[self.train_network.distorted_tau[0],
                                 self.train_network.q_dist_distorted[0]],
                        feed_dict=feed_dict)

        # print(h.shape)

        from matplotlib import pyplot as plt
        # plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=2)
        # # plt.subplot(len(self.train_network.actions), 2, [1, 3])
        # from scipy.misc import imresize
        # plt.imshow(inp[:, :, 0], interpolation="nearest", cmap="gray")

        for i in range(h.shape[0]):
            plt.subplot2grid((2, 1), (i, 0), colspan=1, rowspan=1)
            plt.stem(tau, h[i], markerfmt=" ")

        plt.pause(0.1)
        plt.gcf().clear()

        data = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

        return data

    def y(self, x):
        return self.sess.run(self.train_network.q,
                             feed_dict={self.train_network_base.x: x,
                                        self.train_network.num_samples: self.cfg["K"]})
