import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorboard.plugins.beholder import Beholder

import params


class C51Agent():
    class Model():
        def __init__(self, session, num_actions, train_net):
            self.sess = session

            # Input
            self.x = tf.placeholder(name="state",
                                    dtype=tf.uint8,
                                    shape=(None, params.STATE_DIMENSIONS[0],
                                           params.STATE_DIMENSIONS[1],
                                           params.HISTORY_LEN))

            self.normalized_x = tf.cast(self.x, dtype=tf.float32) / 255.0

            with tf.variable_scope("common"):
                # Convolutional Layers
                self.conv_outputs = []
                for CONV_LAYER_SPEC in params.CONVOLUTIONAL_LAYERS_SPEC:
                    self.conv_outputs.append(
                    tf.layers.conv2d(
                        name="conv_layer_" + str(len(self.conv_outputs) + 1),
                        inputs=self.normalized_x if len(self.conv_outputs) == 0 else
                               self.conv_outputs[-1],
                        filters=CONV_LAYER_SPEC["filters"],
                        kernel_size=CONV_LAYER_SPEC["kernel_size"],
                        strides=CONV_LAYER_SPEC["strides"],
                        activation=tf.nn.relu
                        )
                    )

                # Flatten
                self.flattened_conv_output = tf.layers.flatten(
                    name="conv_output_flattener", inputs=self.conv_outputs[-1]
                )

                # Hidden Layer
                self.dense_outputs = []
                for DENSE_LAYER_SPEC in params.DENSE_LAYERS_SPEC:
                    self.dense_outputs.append(
                        tf.layers.dense(
                            name="dense_layer_" + str(len(self.dense_outputs) + 1),
                            inputs=self.flattened_conv_output if
                                len(self.dense_outputs) == 0 else
                                self.dense_outputs[-1],
                            units=DENSE_LAYER_SPEC, activation=tf.nn.relu
                        )
                    )

                # State-Action-Value Distributions (as a flattened vector)
                self.flattened_q_dist = tf.layers.dense(
                    name="flattened_action_value_dist_logits", inputs=self.dense_outputs[-1],
                    units=num_actions * params.NB_ATOMS
                )

                # Unflatten
                self.q_dist_logits = tf.reshape(
                    self.flattened_q_dist, [-1, num_actions,
                                            params.NB_ATOMS],
                    name="reshape_q_dist_logits"
                )

                # Softmax State-Action-Value Distributions (per action)
                self.q_dist = tf.nn.softmax(
                    self.q_dist_logits, name="action_value_dist", axis=-1
                )

                # Multiply bin probabilities by value
                self.delta_z = (params.V_MAX - params.V_MIN) / (params.NB_ATOMS - 1)
                self.Z = tf.range(start=params.V_MIN, limit=params.V_MAX + self.delta_z,
                                  delta=self.delta_z)
                self.post_mul = self.q_dist * tf.reshape(self.Z, [1, 1, params.NB_ATOMS])

                # Take sum to get the expected state-action values for each action
                self.actions = tf.reduce_sum(self.post_mul, axis=2)

                self.batch_size_range = tf.range(start=0, limit=tf.shape(self.x)[0])

            if not train_net:
                self.targ_q_net_max = tf.summary.scalar("targ_q_net_max", tf.reduce_max(self.actions))
                self.targ_q_net_mean = tf.summary.scalar("targ_q_net_mean", tf.reduce_mean(self.actions))
                self.targ_q_net_min = tf.summary.scalar("targ_q_net_min", tf.reduce_min(self.actions))

                # Find argmax action given expected state-action values at next state
                self.argmax_action = tf.argmax(self.actions, axis=-1,
                                               output_type=tf.int32)

                # Get it's corresponding distribution (this is the target distribution)
                self.argmax_action_distribution = tf.gather_nd(
                    self.q_dist,
                    tf.stack(
                        (self.batch_size_range, self.argmax_action),
                        axis=1
                    )
                )  # Axis = 1 => [N, 2]

                self.mean_argmax_next_state_value = tf.summary.scalar("mean_argmax_q_target",
                      tf.reduce_mean(self.Z * self.argmax_action_distribution))

                # Placeholder for reward
                self.r = tf.placeholder(name="reward", dtype=tf.float32, shape=(None,))
                self.t = tf.placeholder(name="terminal", dtype=tf.uint8, shape=(None,))

                # Compute Tz (Bellman Operator) on atom of expected state-action-value
                # r + gamma * z clipped to [V_min, V_max]
                self.Tz = tf.clip_by_value(tf.reshape(self.r, [-1, 1]) + 0.99 *
                    tf.cast(tf.reshape(self.t, [-1, 1]), tf.float32) * self.Z,
                    clip_value_min=params.V_MIN, clip_value_max=params.V_MAX)

                # Compute bin number (will be floating point).
                self.b = (self.Tz - params.V_MIN)/self.delta_z

                # Lower and Upper Bins.
                self.l = tf.floor(self.b)
                self.u = tf.ceil(self.b)

                # Add weight to the lower bin based on distance from upper bin to
                # approximate bin index b. (0--b--1. If b = 0.3. Then, assign bin
                # 0, p(b) * 0.7 weight and bin 1, p(Z = z_b) * 0.3 weight.)
                self.indexable_l = tf.stack(
                    (
                        tf.reshape(self.batch_size_range, [-1, 1]) *
                        tf.ones((1, params.NB_ATOMS), dtype=tf.int32),
                        # BATCH_SIZE_RANGE x NB_ATOMS [[0, ...], [1, ...], ...]
                        tf.cast(self.l, dtype=tf.int32)
                    ), axis=-1
                )
                self.m_l_vals = self.argmax_action_distribution * (self.u - self.b)
                self.m_l = tf.scatter_nd(tf.reshape(self.indexable_l, [-1, 2]),
                                         tf.reshape(self.m_l_vals, [-1]),
                                         tf.shape(self.l))

                # Add weight to the lower bin based on distance from upper bin to
                # approximate bin index b.
                self.indexable_u = tf.stack(
                    (
                        tf.reshape(self.batch_size_range, [-1, 1]) *
                        tf.ones((1, params.NB_ATOMS), dtype=tf.int32),
                        # BATCH_SIZE_RANGE x NB_ATOMS [[0, ...], [1, ...], ...]
                        tf.cast(self.u, dtype=tf.int32)
                    ), axis=-1
                )
                self.m_u_vals = self.argmax_action_distribution * (self.b - self.l)
                self.m_u = tf.scatter_nd(tf.reshape(self.indexable_u, [-1, 2]),
                                         tf.reshape(self.m_u_vals, [-1]),
                                         tf.shape(self.u))

                # Add Contributions of both upper and lower parts and
                # stop gradient to not update the target network.
                self.m = tf.stop_gradient(tf.squeeze(self.m_l + self.m_u))

                self.weighted_m = tf.clip_by_value(self.m * self.Z,
                     clip_value_min=params.V_MIN, clip_value_max=params.V_MAX)

                self.weighted_m_mean = tf.summary.scalar("mean_q_target",
                     tf.reduce_mean(self.weighted_m))

                self.targ_dist = tf.summary.histogram("target_distribution",
                                     self.weighted_m)

                self.targn_summary = tf.summary.merge([self.targ_dist, self.weighted_m_mean, self.targ_q_net_max,
                                                                            self.targ_q_net_mean,
                                                                            self.targ_q_net_min,
                                                       self.mean_argmax_next_state_value])
            else:
                self.trn_q_net_max = tf.summary.scalar("trn_q_net_max", tf.reduce_max(self.actions))
                self.trn_q_net_mean = tf.summary.scalar("trn_q_net_mean", tf.reduce_mean(self.actions))
                self.trn_q_net_min = tf.summary.scalar("trn_q_net_min", tf.reduce_min(self.actions))

                # Given you took this action.
                self.action_placeholder = tf.placeholder(name="action", dtype=tf.int32, shape=[None, ])

                # Compute Q-Dist. for the action.
                self.action_q_dist = tf.gather_nd(self.q_dist,
                                                  tf.stack((self.batch_size_range,
                                                            self.action_placeholder),
                                                           axis=1))

                self.weighted_q_dist = tf.clip_by_value(self.action_q_dist * self.Z,
                    clip_value_min=params.V_MIN, clip_value_max=params.V_MAX)

                tnd_summary = tf.summary.histogram("training_net_distribution",
                                                   self.weighted_q_dist)

                tnd_mean_summary = tf.summary.scalar("training_net_distribution_mean", tf.reduce_mean(self.weighted_q_dist))

                # Get target distribution.
                self.m_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, params.NB_ATOMS), name="m_placeholder")
                self.loss_sum = -tf.reduce_sum(self.m_placeholder *
                                           tf.log(self.action_q_dist + 1e-5), axis=-1)

                self.loss = tf.reduce_mean(self.loss_sum)

                l_summary = tf.summary.scalar("loss", self.loss)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=params.LEARNING_RATE, epsilon=params.EPSILON_ADAM)
                gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
                grad_norm_summary = tf.summary.histogram("grad_norm", tf.global_norm(gradients))
                gradients, _ = tf.clip_by_global_norm(gradients, params.GRAD_NORM_CLIP)
                self.train_step = self.optimizer.apply_gradients(zip(gradients, variables))
                self.trnn_summary = tf.summary.merge([tnd_mean_summary, tnd_summary, l_summary, grad_norm_summary, self.trn_q_net_max,
                                                                            self.trn_q_net_mean,
                                                                            self.trn_q_net_min])

    def __init__(self):
        self.num_actions = len(params.GLOBAL_MANAGER.actions)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.experience_replay = deque(maxlen=params.EXPERIENCE_REPLAY_SIZE)
        with tf.variable_scope("train_net"):
            self.train_net = self.Model(self.sess, num_actions=self.num_actions, train_net=True)
        with tf.variable_scope("target_net"):
            self.target_net = self.Model(self.sess, num_actions=self.num_actions, train_net=False)
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(params.TENSORBOARD_FOLDER)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        main_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_net/common')
        target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net/common')

        # I am assuming get_collection returns variables in the same order, please double
        # check this is actually happening

        assign_ops = []
        for main_var, target_var in zip(sorted(main_variables, key=lambda x : x.name),
                                        sorted(target_variables, key=lambda x: x.name)):
            assert(main_var.name.replace("train_net", "") == target_var.name.replace("target_net", ""))
            assign_ops.append(tf.assign(target_var, main_var))

        self.copy_operation = tf.group(*assign_ops)

        self.saver = tf.train.Saver(max_to_keep=params.MAX_MODELS_TO_KEEP,
            keep_checkpoint_every_n_hours=params.MIN_MODELS_EVERY_N_HOURS)
        # self.profiler = tf.profiler.Profiler(self.sess.graph)

        self.beholder = Beholder(params.TENSORBOARD_FOLDER)

    def act(self, x):
        if np.random.random() < params.EPSILON_START - \
                (params.GLOBAL_MANAGER.timestep / params.EPSILON_FINAL_STEP) * \
                (1 - params.EPSILON_END):
            return np.random.randint(0, self.num_actions)
        else:
            actions = self.sess.run(fetches=self.train_net.actions,
                                    feed_dict={self.train_net.x: x})
            return np.argmax(actions)

    def add(self, x, a, r, x_p, t):
        assert(np.issubdtype(x.dtype, np.integer))
        self.experience_replay.appendleft([x, a, r, x_p, not t])

    def update(self, x, a, r, x_p, t):
        self.add(x, a, r, x_p, t)

        total_loss = 0
        batch_data = random.sample(self.experience_replay, 32)
        batch_x = np.array([i[0] for i in batch_data])
        batch_a = [i[1] for i in batch_data]
        batch_x_p = np.array([np.array(np.dstack((i[0][:, :, 1:], np.maximum(i[3], i[0][:, :, 3]))))
                             for i in batch_data])
        batch_r = [i[2] for i in batch_data]
        batch_t = [i[4] for i in batch_data]

        targn_summary, m, Tz, b, u, l, indexable_u, indexable_l, m_u_vals, m_l_vals, m_u, m_l = self.sess.run([self.target_net.targn_summary, self.target_net.m,
                                          self.target_net.Tz, self.target_net.b,
                                          self.target_net.u, self.target_net.l,
                                          self.target_net.indexable_u, self.target_net.indexable_l,
                                          self.target_net.m_u_vals, self.target_net.m_l_vals, self.target_net.m_u, self.target_net.m_l], feed_dict={self.target_net.x: batch_x_p,
                                                        self.target_net.r: batch_r, self.target_net.t: batch_t})

        trnn_summary, loss, _ = self.sess.run([self.train_net.trnn_summary, self.train_net.loss, self.train_net.train_step],
                                feed_dict={self.train_net.x: batch_x,
                                           self.train_net.action_placeholder:
                                               batch_a,
                                           self.train_net.m_placeholder: m})

        self.writer.add_summary(targn_summary, params.GLOBAL_MANAGER.num_updates)
        self.writer.add_summary(trnn_summary, params.GLOBAL_MANAGER.num_updates)

        total_loss += loss

        self.beholder.update(self.sess, frame=batch_x[0], arrays=[m, Tz, b, u, l, indexable_u, indexable_l, m_u_vals, m_l_vals, m_u, m_l])

        if params.GLOBAL_MANAGER.num_updates > 0 and \
                params.GLOBAL_MANAGER.num_updates % params.COPY_TARGET_FREQ == 0:
            self.sess.run(self.copy_operation)
            print("Copied to target. Current Loss: ", total_loss)

        if params.GLOBAL_MANAGER.num_updates > 0 and \
                params.GLOBAL_MANAGER.num_updates % params.MODEL_SAVE_FREQ == 0:
            self.saver.save(self.sess, params.MODELS_FOLDER,
                            global_step=params.GLOBAL_MANAGER.num_updates,
                            write_meta_graph=(params.GLOBAL_MANAGER.num_updates <=
                                              params.MODEL_SAVE_FREQ))