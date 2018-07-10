import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import agent
from configuration import ConfigurationManager
from function_approximator.head import FixedAtomsDistributionalHead


class QuantileRegressionAgent(agent.DistributionalAgent):
    required_params = ["COPY_TARGET_FREQUENCY",
                       "UPDATE_FREQUENCY", "DISCOUNT_FACTOR"]

    def __init__(self, cfg_parser: ConfigurationManager):
        super().__init__(cfg_parser)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 16
        config.inter_op_parallelism_threads = 16
        self.sess = tf.Session(config=config)

        self.cfg_parser = cfg_parser

        self.num_updates = 0

        self.cfg = cfg_parser.parse_and_return_dictionary(
            "AGENT", QuantileRegressionAgent.required_params)

        from function_approximator import GeneralNetwork
        with tf.variable_scope("train_net"):
            self.train_network_base = GeneralNetwork(cfg_parser)
            self.train_network = FixedAtomsDistributionalHead(
                cfg_parser, self.train_network_base)
        with tf.variable_scope("target_net"):
            self.target_network_base = GeneralNetwork(cfg_parser)
            self.target_network = FixedAtomsDistributionalHead(
                cfg_parser, self.target_network_base)

        from util.util import get_copy_op
        self.copy_operation = get_copy_op("train_net",
                                          "target_net")

        from memory.experience_replay import ExperienceReplay
        self.experience_replay = ExperienceReplay(cfg_parser)

        self.build_networks()

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.sess.run(self.copy_operation)

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

    def learn(self, experiences):
        batch_x = np.array([i[0] for i in experiences])
        batch_a = [i[1] for i in experiences]
        batch_x_p = np.array([i[3] for i in experiences])
        batch_r = [i[2] for i in experiences]
        batch_t = [i[4] for i in experiences]
        #print("batch rat", len(batch_r), batch_r)
        #print("batch rat", len(batch_a), batch_a)
        #print("batch rat", len(batch_t), batch_t)
        #print( batch_r.shape, batch_a.shape, batch_t.shape)

        return self.sess.run([self.train_step],
                             feed_dict={self.train_network_base.x: batch_x,
                                        self.action_placeholder: batch_a,
                                        self.target_network_base.x: batch_x_p,
                                        self.r: batch_r,
                                        self.t: batch_t})

    def act(self, x):
        if random.random() < 1.0 - (min(10000, self.num_updates) / 10000) * (1 - 0.1):
            return [self.train_network.act_to_send(
                random.choice(self.train_network.actions)
            )]
        else:
            return [self.train_network.act_to_send(self.greedy_action([x])[0])]

    def viz_dist(self, x):
        # Plot
        h = np.squeeze(self.sess.run(fetches=self.train_network.y,
                       feed_dict={self.train_network_base.x: x}))
        l, s = np.linspace(self.cfg["V_MIN"],
                           self.cfg["V_MAX"],
                           self.train_network.cfg["NB_ATOMS"],
                           retstep=True)

        for i in range(h.shape[0]):
            plt.subplot(len(self.train_network.actions), 1, i + 1)
            plt.bar(l - s/2., height=h[i], width=s,
                    color="brown", edgecolor="red", linewidth=0.5, align="edge")

        plt.pause(0.1)
        plt.gcf().clear()

    def add(self, x, a, r, x_p, t):
        self.experience_replay.add([x, a, r, x_p, not t])

    def update(self, x, a, r, x_p, t):
        self.num_updates += 1
        self.add(x, a, r, x_p, t)

        if self.experience_replay.size() > self.cfg["MINIBATCH_SIZE"]:
            self.learn(self.experience_replay.sample(self.cfg["MINIBATCH_SIZE"]))

        if self.num_updates > 0 and \
            self.num_updates % self.cfg["COPY_TARGET_FREQ"] == 0:
            self.sess.run(self.copy_operation)
            print("Copied.")
            assert(np.allclose(self.sess.run(self.train_network.y, feed_dict={self.train_network_base.x: [x]}),
                   self.sess.run(self.target_network.y, feed_dict={self.target_network_base.x: [x]})))

