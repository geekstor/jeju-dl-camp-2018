import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import agent
from configuration import ConfigurationManager
from function_approximator.head import SoftmaxFixedAtomsDistributionalHead


class CategoricalAgent(agent.DistributionalAgent):
    required_params = ["UPDATE_FREQUENCY", "DISCOUNT_FACTOR",
                       "V_MIN", "V_MAX"]

    head = SoftmaxFixedAtomsDistributionalHead

    def __init__(self, cfg_parser: ConfigurationManager):
        super().__init__(cfg_parser, CategoricalAgent.head)
        self.cfg = cfg_parser.parse_and_return_dictionary(
            "AGENT", CategoricalAgent.required_params)

        self.build_loss_op()

        self.prepare(self.loss)

    def build_loss_op(self):
        Z, delta_z = np.linspace(self.cfg["V_MIN"], self.cfg["V_MAX"],
                                 self.train_network.cfg["NB_ATOMS"],
                                 retstep=True)
        Z = tf.constant(Z, dtype=tf.float32, name="Z")
        delta_z = tf.constant(delta_z, dtype=tf.float32, name="Z_step")

        for var_scope, graph in {"train_net": self.train_network,
                                 "target_net": self.target_network}.items():
            with tf.variable_scope(var_scope):
                graph.post_mul = tf.reduce_sum(graph.y * Z,
                                               axis=-1)

                graph.argmax_action = tf.argmax(graph.post_mul, axis=-1,
                                                output_type=tf.int32,
                                                name="argmax_action")

        batch_size_range = tf.range(tf.shape(self.target_network_base.x)[0])
        # Get it's corresponding distribution (this is used for
        # computing the target distribution)
        self.argmax_action_distribution = tf.gather_nd(
            self.target_network.y,
            tf.stack(
                (batch_size_range, self.target_network.argmax_action),
                axis=1, name="index_for_argmax_action_q_dist"
            ), name="gather_nd_for_batch_argmax_action_q_dists"
        )  # Axis = 1 => [N, 2]

        self.Tz = tf.clip_by_value(tf.reshape(self.reward_placeholder, [-1, 1]) + self.cfg["DISCOUNT_FACTOR"] *
                                   tf.cast(tf.reshape(self.terminal_placeholder, [-1, 1]), tf.float32) * Z,
                                   clip_value_min=self.cfg["V_MIN"],
                                   clip_value_max=self.cfg["V_MAX"],
                                   name="Tz")

        # Compute bin number (can be floating point/integer).
        self.b = tf.identity((self.Tz - self.cfg["V_MIN"]) / delta_z, name="b")

        # Lower and Upper Bins.
        self.l = tf.floor(self.b, name="l")
        self.u = tf.ceil(self.b, name="u")

        # Add weight to the lower bin based on distance from upper bin to
        # approximate bin index b. (0--b--1. If b = 0.3. Then, assign bin
        # 0, p(b) * 0.7 weight and bin 1, p(Z = z_b) * 0.3 weight.)
        self.indexable_l = tf.stack(
            (
                tf.identity(tf.reshape(batch_size_range, [-1, 1]) *
                            tf.ones((1, tf.shape(self.target_network.y)[-1]),
                                    dtype=tf.int32), name="index_for_l"),
                # BATCH_SIZE_RANGE x NB_ATOMS [[0, ...], [1, ...], ...]
                tf.cast(self.l, dtype=tf.int32)
            ), axis=-1, name="indexable_l"
        )
        self.m_l_vals = tf.identity(self.argmax_action_distribution * (1 - (self.b - self.l)),
                                    name="values_to_add_for_m_l")
        self.m_l = tf.scatter_nd(tf.reshape(self.indexable_l, [-1, 2]),
                                 tf.reshape(self.m_l_vals, [-1]),
                                 tf.shape(self.l), name="m_l")

        # Add weight to the lower bin based on distance from upper bin to
        # approximate bin index b.
        self.indexable_u = tf.stack(
            (
                tf.identity(tf.reshape(batch_size_range, [-1, 1]) *
                            tf.ones((1, tf.shape(self.target_network.y)[-1]),
                                    dtype=tf.int32), name="index_for_u"),
                # BATCH_SIZE_RANGE x NB_ATOMS [[0, ...], [1, ...], ...]
                tf.cast(self.u, dtype=tf.int32)
            ), axis=-1, name="indexable_u"
        )
        self.m_u_vals = tf.identity(self.argmax_action_distribution * (self.b - self.l),
                                    name="values_to_add_for_m_u")
        self.m_u = tf.scatter_nd(tf.reshape(self.indexable_u, [-1, 2]),
                                 tf.reshape(self.m_u_vals, [-1]),
                                 tf.shape(self.u), name="m_u")

        # Add Contributions of both upper and lower parts and
        # stop gradient to not update the target network.
        self.m = tf.stop_gradient(tf.squeeze(self.m_l + self.m_u, name="m"))

        batch_size_range = tf.range(start=0,
                                        limit=tf.shape(self.train_network_base.x)[0])

        # Compute Q-Dist. for the action.
        self.action_q_dist = tf.gather_nd(self.train_network.y,
                                          tf.stack((batch_size_range,
                                                    self.action_placeholder),
                                                   axis=1))

        self.loss_sum = -tf.reduce_sum(self.m *
                                       tf.log(self.action_q_dist + 1e-5), axis=-1, name="loss")

        self.loss = tf.reduce_mean(self.loss_sum)

    def distribution(self, state):
        return self.sess.run(fetches=[self.train_network.y],
                             feed_dict={self.train_network_base.x: state})

    def greedy_action(self, state):
        return self.sess.run(fetches=self.train_network.argmax_action,
                             feed_dict={self.train_network_base.x: state})

    def act(self, x):
        greedy_f = lambda inp: self.greedy_action([inp])[0]
        exploratory_f = lambda inp: random.randint(0, self.cfg_parser["NUM_ACTIONS"] - 1)

        if random.random() < 1.0 - (min(10000, self.num_updates) / 10000) * (1 - 0.1):
            return random.randint(0, self.cfg_parser["NUM_ACTIONS"] - 1)
        else:
            return self.greedy_action([x])[0]
