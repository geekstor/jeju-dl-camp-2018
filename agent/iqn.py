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

        self.loss = self.build_loss_op()

        self.prepare(self.loss)

    def build_loss_op(self) -> tf.Operation:
        batch_dim_range = tf.range(tf.shape(self.train_network_base.x)[0],
                                   dtype=tf.int32)

        # OPS. for computing dist. from current train net
        flat_indices_chosen_actions = tf.stack([batch_dim_range, self.action_placeholder],
                                               axis=1)
        dist_of_chosen_actions = tf.gather_nd(self.train_network.q_dist,
                                              flat_indices_chosen_actions)

        # OPS. for computing target quantile dist.
        flat_indices_for_argmax_action_target_net = tf.stack([
            batch_dim_range, self.target_network.greedy_action], axis=1)

        sampled_return_of_greedy_actions_target_net = \
            tf.gather_nd(self.target_network.q_dist,
                         flat_indices_for_argmax_action_target_net)

        expected_quantiles = self.reward_placeholder[:, tf.newaxis] + \
            self.cfg["DISCOUNT_FACTOR"] * tf.cast(
            self.terminal_placeholder[:, tf.newaxis], dtype=tf.float32) * \
            sampled_return_of_greedy_actions_target_net

        u = expected_quantiles[:, tf.newaxis, :] - \
            dist_of_chosen_actions[:, :, tf.newaxis]
        k = 1
        huber_loss = 0.5 * tf.square(tf.clip_by_value(tf.abs(u), 0.0, k))
        huber_loss += k * (tf.abs(u) - tf.clip_by_value(tf.abs(u), 0.0, k))
        quantile_loss = tf.abs(tf.reshape(tf.tile(self.train_network.tau, [1, tf.shape(
            self.target_network.tau)[1]]), [-1, tf.shape(self.train_network.tau)[1],
                                            tf.shape(self.target_network.tau)[1]]) -
            tf.cast((u < 0), tf.float32)) * huber_loss

        loss = tf.reduce_mean(quantile_loss)

        diversity_bonus = -tf.nn.moments(self.train_network.tau, axes=-1)[1]

        return loss #+ diversity_bonus

    def greedy_action(self, state):
        return self.sess.run(fetches=self.train_network.greedy_action,
                             feed_dict={self.train_network_base.x: state,
                                        self.train_network.num_samples: self.cfg["K"]})

    def act(self, x) -> int:
        if random.random() < 1.0 - (min(10000, self.predict_calls) / 10000) * (1 - 0.1):
            return random.randint(0, self.cfg_parser["NUM_ACTIONS"] - 1)
        else:
            return self.greedy_action([x])

    def learn(self, experiences):
        feed_dict = self.batch_experiences(experiences)
        feed_dict.update({
            self.train_network.num_samples: self.cfg["N"],
            self.target_network.num_samples: self.cfg["N_PRIME"]
        })
        loss, _, o_q, t_q  = self.sess.run(fetches=[self.loss, self.train_step,
                                         self.target_network.q_dist,
                                         self.train_network.q_dist], feed_dict=feed_dict)
        #print(loss, "Online", o_q[0], "Target", t_q[0])