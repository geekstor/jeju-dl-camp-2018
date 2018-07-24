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

        self.cfg["NB_ATOMS"] = self.cfg_parser["HEAD.NB_ATOMS"]

        self.Z, self.delta_z = np.linspace(self.cfg["V_MIN"], self.cfg["V_MAX"],
                                 self.cfg["NB_ATOMS"],
                                 retstep=True)

        self.loss = self.build_loss_op()

        self.prepare(self.loss)

    def build_loss_op(self):
        Z = tf.constant(self.Z, dtype=tf.float32, name="Z")
        delta_z = tf.constant(self.delta_z, dtype=tf.float32, name="Z_step")

        Z_batch = tf.reshape(tf.tile(Z, [tf.shape(self.train_network_base.x)[0]]),
                             [tf.shape(self.train_network_base.x)[0], self.cfg["NB_ATOMS"]])

        Tz = tf.clip_by_value(self.bellman_op(Z_batch),
                              self.cfg["V_MIN"], self.cfg["V_MAX"])

        tiled_Tz = tf.reshape(tf.tile(Tz, [1, self.cfg["NB_ATOMS"]]),
                              [-1, self.cfg["NB_ATOMS"], self.cfg["NB_ATOMS"]])
        tiled_Z = tf.reshape(tf.tile(Z_batch, [1, self.cfg["NB_ATOMS"]]),
                             [-1, self.cfg["NB_ATOMS"], self.cfg["NB_ATOMS"]])

        inner_term = tf.abs(tiled_Tz - tf.transpose(tiled_Z, [0, 2, 1])) / delta_z

        left_term = tf.clip_by_value(1 - inner_term, 0, 1)

        right_term = self.dist_of_target_max_q

        projected_update = tf.einsum("ijk,ik->ij", left_term, right_term)

        loss = -1 * projected_update * tf.log(self.dist_of_chosen_actions)

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def y(self, state):
        return np.sum(self.sess.run(fetches=[self.train_network.q_dist],
                                    feed_dict={self.train_network_base.x: state}) * self.Z,
                      axis=-1)