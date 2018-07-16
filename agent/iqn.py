import agent
import random
import numpy as np
import tensorflow as tf
from configuration import ConfigurationManager
from function_approximator.head import IQNHead


class ImplicitQuantileAgent(agent.DistributionalAgent):
    required_params = ["UPDATE_FREQUENCY", "DISCOUNT_FACTOR",
                       "N", # Number of Samples of Tau to take when getting Q-value (online net)
                       "N_PRIME", # Number of Samples of Tau to take when calculating target
                       "K"] # Number of Samples of Tau to take when acting # TODO: Beta

    head = IQNHead

    def __init__(self, cfg_parser: ConfigurationManager):
        super().__init__(cfg_parser, ImplicitQuantileAgent.head)

        self.build_loss_and_train_ops()

        self.prepare()

    def build_loss_and_train_ops(self):
        self.batch_dim_range = tf.range(tf.shape(self.train_network_base.x)[0],
                                                dtype=tf.int32)

        # OPS. for computing dist. from current train net
        self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.flat_indices_chosen_actions = tf.stack([self.batch_dim_range,
                                                     self.action_placeholder],
                                                     axis=1)
        self.dist_of_chosen_actions = tf.gather_nd(self.train_network.q_dist,
                                                   self.flat_indices_chosen_actions)

        # OPS. for computing target quantile dist.
        self.flat_indices_for_argmax_action_target_net = tf.stack([self.batch_dim_range,
                                                                   self.target_network.greedy_action], axis=1)

        self.sampled_return_of_greedy_actions_target_net = \
            tf.gather_nd(self.target_network.q_dist,
                         self.flat_indices_for_argmax_action_target_net)

        self.r = tf.placeholder(shape=[None,], dtype=tf.float32)
        self.t = tf.placeholder(shape=[None,], dtype=tf.uint8)

        self.expected_quantiles = self.r[:, tf.newaxis] + self.cfg["DISCOUNT_FACTOR"] * \
                                  tf.cast(self.t[:, tf.newaxis], dtype=tf.float32) * \
                                  self.sampled_return_of_greedy_actions_target_net

        u = self.expected_quantiles[:, tf.newaxis, :] - self.dist_of_chosen_actions[:, :, tf.newaxis]
        k = 1
        huber_loss = 0.5 * tf.square(tf.clip_by_value(tf.abs(u), 0.0, k))
        huber_loss += k * (tf.abs(u) - tf.clip_by_value(tf.abs(u), 0.0, k))
        quantile_loss = tf.abs(tf.reshape(tf.tile(self.train_network.tau, [1, tf.shape(
            self.target_network.tau)[1]]), [-1, tf.shape(self.train_network.tau)[1],
                                        tf.shape(self.target_network.tau)[1]]) - tf.cast((u < 0), tf.float32)) * \
                        huber_loss
        self.loss = tf.reduce_mean(quantile_loss)

        from optimizer.optimizer import get_optimizer
        self.train_step = get_optimizer(self.cfg_parser, self.loss,
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                              scope='train_net'))

    def greedy_action(self, state):
        return self.sess.run(fetches=self.train_network.greedy_action,
                             feed_dict={self.train_network_base.x: state,
                                        self.train_network.num_samples: self.cfg["K"]})

    def act(self, x):
        if random.random() < 1.0 - (min(10000, self.num_updates) / 10000) * (1 - 0.1):
            return random.randint(0, self.train_network.num_actions - 1)
        else:
            return self.greedy_action([x])

    def learn(self, experiences):
        batch_x = np.array([i[0] for i in experiences])
        batch_a = [i[1] for i in experiences]
        batch_x_p = np.array([i[3] for i in experiences])
        batch_r = [i[2] for i in experiences]
        batch_t = [i[4] for i in experiences]

        return self.sess.run([self.train_step],
                             feed_dict={self.train_network_base.x: batch_x,
                                        self.action_placeholder: batch_a,
                                        self.target_network_base.x: batch_x_p,
                                        self.r: batch_r,
                                        self.t: batch_t,
                                        self.train_network.num_samples: self.cfg["N"],
                                        self.target_network.num_samples: self.cfg["N_PRIME"]})