import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboard.plugins.beholder import Beholder

from agent import DistributionalAgent
from configuration import ConfigurationManager


class CategoricalAgent(DistributionalAgent):
    required_params = ["EXPERIENCE_REPLAY_SIZE", "COPY_TARGET_FREQUENCY",
                       "UPDATE_FREQUENCY", "DISCOUNT_FACTOR", "V_MIN", "V_MAX"]

    def distribution(self):
        pass

    def greedy_action(self, state):
        return super().greedy_action(state)

    def learn(self, experiences):
        return super().learn(experiences)

    def __init__(self, cfg_parser: ConfigurationManager, num_actions, observation_dims):
        super().__init__(cfg_parser)

        self.cfg = cfg_parser.parse_and_return_dictionary(
            "AGENT", CategoricalAgent.required_params)

        from function_approximator import GeneralNetwork
        with tf.variable_scope("train_net/common"):
            self.train_network = GeneralNetwork(cfg_parser, num_actions, observation_dims, True)
        with tf.variable_scope("target_net/common"):
            self.target_network = GeneralNetwork(cfg_parser, num_actions, observation_dims)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 16
        config.inter_op_parallelism_threads = 16
        self.sess = tf.Session(config=config)

        self.experience_replay = deque(maxlen=self.cfg["EXPERIENCE_REPLAY_SIZE"])
        self.build_networks()
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(params.TENSORBOARD_FOLDER)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        train_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_net/common')
        target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/common')

        assign_ops = []
        for main_var, target_var in zip(sorted(train_variables, key=lambda x : x.name),
                                        sorted(target_variables, key=lambda x: x.name)):
            if(main_var.name.replace("train_net", "") == target_var.name.replace("target_net", "")):
                assign_ops.append(tf.assign(target_var, main_var))

        self.copy_operation = tf.group(*assign_ops)

        self.sess.run(self.copy_operation)

        self.saver = tf.train.Saver(max_to_keep=params.MAX_MODELS_TO_KEEP,
                                    keep_checkpoint_every_n_hours=params.MIN_MODELS_EVERY_N_HOURS)

        self.beholder = Beholder(params.TENSORBOARD_FOLDER)


    def build_networks(self):
        z, delta_z = np.linspace(self.cfg["V_MIN"], self.cfg["V_MAX"],
                                 self.train_network.cfg["NB_ATOMS"],
                                 retstep=True)
        self.Z = tf.constant(z, dtype=tf.float32, name="Z")
        self.delta_z = tf.constant(delta_z, dtype=tf.float32, name="Z_step")

        for var_scope, graph in {"train_net": self.train_network,
                                 "target_net": self.target_network}.items():
            with tf.variable_scope(var_scope):
                # State-Action-Value Distributions (per action) using logits
                graph.q_dist = tf.nn.softmax(
                    graph.y, name="state_action_value_dist", axis=-1
                )

                graph.post_mul = tf.reduce_sum(graph.y * self.Z, axis=-1)

                # Take sum to get the expected state-action values for each action
                # graph.actions = tf.reduce_sum(graph.post_mul, axis=2,
                #                               name="expected_state_action_value")

                graph.argmax_action = tf.argmax(graph.post_mul, axis=-1,
                                                output_type=tf.int32,
                                                name="argmax_action")

        obj = self.target_network
        with tf.variable_scope("target_net"):
            # Find argmax action given expected state-action values at next state
            obj.batch_size_range = tf.range(start=0, limit=tf.shape(obj.x)[0])

            # Get it's corresponding distribution (this is used for
            # computing the target distribution)
            cat_idx = tf.transpose(tf.reshape(tf.concat([obj.batch_size_range,
                                                         obj.argmax_action],
                                                        axis=0), [2, tf.shape(obj.x)[0]]))
            p_best = tf.gather_nd(obj.q_dist, cat_idx)

            # Placeholder for reward and terminal
            obj.r = tf.placeholder(name="reward", dtype=tf.float32, shape=(None,))
            obj.t = tf.placeholder(name="terminal", dtype=tf.uint8, shape=(None,))
            # TODO: Optimize memory uint8 -> bool (check if casting works to float)

            big_z = tf.reshape(tf.tile(self.Z, [self.cfg["MINIBATCH_SIZE"]]),
                               [self.cfg["MINIBATCH_SIZE"],
                                self.train_network.cfg["NB_ATOMS"]])
            big_r = tf.transpose(tf.reshape(tf.tile(obj.r, [self.train_network.cfg["NB_ATOMS"]]),
                                            [self.train_network.cfg["NB_AROMS"], self.cfg["MINIBATCH_SIZE"]]))

            # Compute Tz (Bellman Operator) on atom of expected state-action-value
            # r + gamma * z clipped to [V_min, V_max]
            obj.Tz = tf.clip_by_value(big_r + params.DISCOUNT_FACTOR *
                                      tf.einsum('ij,i->ij', big_z,
                                                tf.cast(obj.t, tf.float32)),
                                      self.cfg["V_MIN"], self.cfg["V_MAX"])

            big_Tz = tf.reshape(tf.tile(obj.Tz, [1, self.train_network.cfg["NB_ATOMS"]]), [-1, self.train_network.cfg["NB_ATOMS"],
                                                                                          self.train_network.cfg[
                                                                                              "NB_ATOMS"]])
            big_z = tf.reshape(tf.tile(self.Z, [self.cfg["MINIBATCH_SIZE"]]),
                               [self.cfg["MINIBATCH_SIZE"],
                                self.train_network.cfg["NB_ATOMS"]])
            big_big_z = tf.reshape(tf.tile(big_z, [1, self.train_network.cfg["NB_ATOMS"]]),
                                   [-1, self.train_network.cfg["NB_ATOMS"], self.train_network.cfg["NB_ATOMS"]])

            Tzz = tf.abs(big_Tz - tf.transpose(big_big_z, [0, 2, 1])) / self.delta_z
            Thz = tf.clip_by_value(1 - Tzz, 0, 1)

            obj.m = tf.einsum('ijk,ik->ij', Thz, p_best)

        obj = self.train_network
        with tf.variable_scope("train_net"):
            obj.batch_size_range = tf.range(start=0, limit=tf.shape(obj.x)[0])

            # Given you took this action.
            obj.action_placeholder = tf.placeholder(name="action", dtype=tf.int32, shape=[None, ])

            cat_idx = tf.transpose(tf.reshape(tf.concat([obj.batch_size_range,
                                                         obj.action_placeholder], axis=0), [2, tf.shape(obj.x)[0]]))
            p_t_next = tf.gather_nd(obj.q_dist, cat_idx)

            # Get target distribution.
            obj.loss_sum = tf.reduce_sum(-1 * self.target_net.m *
                                       tf.log(p_t_next), axis=-1, name="loss")

            obj.loss = tf.reduce_mean(obj.loss_sum)

            from optimizer.optimizer import get_optimizer
            self.train_step = get_optimizer(None, obj.loss,
                                            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                     scope='train_net/common'))

    def act(self, x):
        pass

    def viz_dist(self, x):
        # Plot
        h = np.squeeze(self.sess.run(fetches=self.train_network.q_dist,
                       feed_dict={self.train_network.x: x}))
        l, s = np.linspace(self.cfg["V_MIN"],
                           self.cfg["V_MAX"],
                           self.cfg["NB_ATOMS"],
                           retstep=True)
        for i in range(h.shape[0]):
            plt.subplot(self.num_actions, 1, i + 1)
            plt.bar(l - s/2., height=h[i], width=s,
                    color="brown", edgecolor="red", linewidth=0.5, align="edge")
        plt.pause(0.1)
        plt.gcf().clear()

    def add(self, x, a, r, x_p, t):
        self.experience_replay.appendleft([x, a, r, x_p, not t])

    def update(self, x, a, r, x_p, t):
        self.add(x, a, r, x_p, t)

        total_loss = 0
        batch_data = random.sample(self.experience_replay, self.cfg["MINIBATCH_SIZE"])
        batch_x = np.array([i[0] for i in batch_data])
        batch_a = [i[1] for i in batch_data]
        batch_x_p = np.array([i[3] for i in batch_data])
        batch_r = [i[2] for i in batch_data]
        batch_t = [i[4] for i in batch_data]

        m, loss, _ = self.sess.run([self.target_network.m, self.train_network.loss, self.train_network.train_step],
                                           feed_dict={self.train_network.x: batch_x,
                                           self.train_network.action_placeholder:
                                               batch_a, self.target_network.x: batch_x_p,
                                            self.target_network.r: batch_r, self.target_network.t: batch_t})

        #self.writer.add_summary(targn_summary, params.GLOBAL_MANAGER.num_updates)
        #self.writer.add_summary(trnn_summary, params.GLOBAL_MANAGER.num_updates)

        total_loss += loss

        self.beholder.update(self.sess, frame=batch_x[0], arrays=[m])

        if self.num_updates > 0 and \
            self.num_updates % self.cfg["COPY_TARGET_FREQ"] == 0:
            self.sess.run(self.copy_operation)
            print("Copied to target. Current Loss: ", total_loss)

        if self.num_updates > 0 and \
                self.num_updates % self.MODEL_SAVE_FREQ == 0:
            self.saver.save(self.sess, params.MODELS_FOLDER + "/Model",
                            global_step=params.GLOBAL_MANAGER.num_updates,
                            write_meta_graph=True)


# # QR-DQN
# u = self.m_placeholder - self.action_q_dist
# tau = [(2*i - 1)/(2*N) for i in range(1, N + 1)]
# kappa = tf.constant(1., dtype=tf.float32)
# one_half_kappa_squared = tf.constant(0.5 * kappa * tf.square(kappa))
# l_kappa = tf.where(tf.less_equal(u, kappa), x=0.5*tf.square(u),
#                    y=kappa * tf.abs(u) - one_half_kappa_squared)
# rho_k_tau = tf.where(tf.less(u, 0.), tf.abs(tau - 1.), tf.abs(tau)) * l_kappa
#
#

