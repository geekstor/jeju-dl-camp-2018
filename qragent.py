import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboard.plugins.beholder import Beholder

from configuration import params_cartpole as params
from function_approximator.network import DistributionalAgentNet


class QRAgent:
    def build_networks(self):
        with tf.variable_scope("train_net/common"):
            self.train_net = DistributionalAgentNet()
        with tf.variable_scope("target_net/common"):
            self.target_net = DistributionalAgentNet()

        for var_scope, graph in {"train_net": self.train_net,
                                 "target_net": self.target_net}.items():
            with tf.variable_scope(var_scope):
                # State-Action-Value Distributions (per action) using logits
                graph.q = tf.reduce_mean(graph.y, axis=-1)

                # Take sum to get the expected state-action values for each action
                # graph.actions = tf.reduce_sum(graph.post_mul, axis=2,
                #                               name="expected_state_action_value")

                graph.argmax_action = tf.argmax(graph.q, axis=-1,
                                                output_type=tf.int32,
                                                name="argmax_action")

        obj = self.target_net
        with tf.variable_scope("target_net", reuse=True):
            obj.batch_size_range = tf.range(start=0, limit=tf.shape(obj.x)[0])

            obj.flat_indices = tf.stack([obj.batch_size_range, obj.argmax_action], axis=1)

            obj.qdist_argmax_action = tf.gather_nd(obj.y, obj.flat_indices)
            obj.qdist_argmax_action.set_shape([None, params.NB_ATOMS])

            # Placeholder for reward and terminal
            obj.r = tf.placeholder(name="reward", dtype=tf.float32, shape=(None,))
            obj.t = tf.placeholder(name="terminal", dtype=tf.uint8, shape=(None,))
            # TODO: Optimize memory uint8 -> bool (check if casting works to float)

            obj.Ttheta = tf.identity(obj.r[:, tf.newaxis] + params.DISCOUNT_FACTOR *
                                     tf.cast(obj.t[:, tf.newaxis], dtype=tf.float32) *
                                     obj.qdist_argmax_action, name='quant_target')

        obj = self.train_net
        with tf.variable_scope("train_net", reuse=True):
            obj.batch_size_range = tf.range(start=0, limit=tf.shape(obj.x)[0])

            obj.action_placeholder = tf.placeholder(name="action",
                                                    dtype=tf.int32, shape=[None, ])

            obj.flat_indices = tf.stack([obj.batch_size_range, obj.action_placeholder],
                                        axis=1)

            obj.action_dist = tf.gather_nd(obj.y, obj.flat_indices)
            obj.action_dist.set_shape([None, params.NB_ATOMS])

            from tensorflow.contrib.framework import argsort
            obj.sorted_action_dist_indices = argsort(obj.action_dist)

        # QR-DQN
        u = self.target_net.Ttheta - self.train_net.action_dist
        negative_indicator = tf.cast(tf.less(u, 0), dtype=tf.float32)
        tau = tf.range(0, params.NB_ATOMS + 1, dtype=tf.float32, name='tau') * \
              1. / params.NB_ATOMS
        tau = tf.identity((tau[:-1] + tau[1:]) / 2, name='tau_hat')
        tau = tf.transpose(tf.tile(tf.expand_dims(tau, 1),
                [1, params.MINIBATCH_SIZE]))
        tau = tf.gather_nd(tau, tf.stack([tf.tile(tf.expand_dims(tf.range(params.MINIBATCH_SIZE), 1),
                                                   [1, params.NB_ATOMS]),
                                          obj.sorted_action_dist_indices], axis=-1))
        _kappa = params.KAPPA
        kappa = tf.constant(_kappa, dtype=tf.float32)
        one_half_kappa_squared = tf.constant(0.5 * _kappa * _kappa, tf.float32)
        if _kappa == 0:
            l_kappa = u
            rho_k_tau = tau - negative_indicator
        else:
            l_kappa = tf.where(tf.less_equal(u, kappa), 0.5 * tf.square(u),
                               kappa * tf.abs(u) - one_half_kappa_squared)
            rho_k_tau = tf.abs(tau - negative_indicator)

        loss = rho_k_tau * l_kappa
        self.loss = tf.reduce_sum(loss) / params.NB_ATOMS

        self.optimizer = tf.train.AdamOptimizer(learning_rate=params.LEARNING_RATE,
                                                epsilon=params.EPSILON_ADAM)
        self.train_step = self.optimizer.minimize(self.loss, var_list=tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='train_net/common'))

    def __init__(self):
        self.num_actions = len(params.GLOBAL_MANAGER.actions)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 16
        config.inter_op_parallelism_threads = 16
        self.sess = tf.Session(config=config)
        self.experience_replay = deque(maxlen=params.EXPERIENCE_REPLAY_SIZE)
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

    def act(self, x):
        if np.random.random() < params.EPSILON_START - \
                (min(params.EPSILON_FINAL_STEP, params.GLOBAL_MANAGER.timestep) /
                 params.EPSILON_FINAL_STEP) * \
                (1 - params.EPSILON_END):
            return np.random.randint(0, self.num_actions)
        else:
            return self.sess.run(fetches=self.train_net.argmax_action,
                                    feed_dict={self.train_net.x: x})[0]

    def viz_dist(self, x):
        # Plot
        h = np.squeeze(self.sess.run(fetches=self.train_net.y,
                       feed_dict={self.train_net.x: x}))
        l, s = np.linspace(params.V_MIN, params.V_MAX, params.NB_ATOMS, retstep=True)
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

        if params.GLOBAL_MANAGER.num_updates > 0 and \
                params.GLOBAL_MANAGER.num_updates % params.COPY_TARGET_FREQ == 0:
            self.sess.run(self.copy_operation)

        if params.GLOBAL_MANAGER.num_updates > 0 and \
                params.GLOBAL_MANAGER.num_updates % params.MODEL_SAVE_FREQ == 0:
            self.saver.save(self.sess, params.MODELS_FOLDER + "/Model",
                            global_step=params.GLOBAL_MANAGER.num_updates,
                            write_meta_graph=True)

        total_loss = 0
        batch_data = random.sample(self.experience_replay, params.MINIBATCH_SIZE)
        batch_x = np.array([i[0] for i in batch_data])
        batch_a = [i[1] for i in batch_data]
        batch_x_p = np.array([i[3] for i in batch_data])
        batch_r = [i[2] for i in batch_data]
        batch_t = [i[4] for i in batch_data]

        loss, _ = self.sess.run([self.loss, self.train_step],
                                           feed_dict={self.train_net.x: batch_x,
                                           self.train_net.action_placeholder:
                                               batch_a, self.target_net.x: batch_x_p,
                                            self.target_net.r: batch_r,
                                                      self.target_net.t: batch_t})

        #self.writer.add_summary(targn_summary, params.GLOBAL_MANAGER.num_updates)
        #self.writer.add_summary(trnn_summary, params.GLOBAL_MANAGER.num_updates)

        total_loss += loss

        self.beholder.update(self.sess)

        #print("Current Loss: ", total_loss)

