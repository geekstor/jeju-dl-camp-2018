import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.python import debug as tf_debug
from tensorboard.plugins.beholder import Beholder

class C51Agent():
    class Model():
        def __init__(self, session, num_actions, train_net):
            self.sess = session

            # Input
            self.x = tf.placeholder(name="state",
                                    dtype=tf.float32,
                                    shape=(None, 42, 42, 4))

            with tf.variable_scope("common"):
                # Convolutional Layers
                self.conv_l1_out = tf.layers.conv2d(name="conv_layer_1",
                                                    inputs=self.x,
                                                    filters=32, kernel_size=3,
                                                    strides=2, activation=tf.nn.relu)
                self.conv_l2_out = tf.layers.conv2d(name="conv_layer_2",
                                                    inputs=self.conv_l1_out,
                                                    filters=32, kernel_size=3,
                                                    strides=2, activation=tf.nn.relu)
                self.conv_l3_out = tf.layers.conv2d(name="conv_layer_3",
                                                    inputs=self.conv_l2_out,
                                                    filters=32, kernel_size=3,
                                                    strides=2, activation=tf.nn.relu)
                self.conv_l4_out = tf.layers.conv2d(name="conv_layer_4",
                                                    inputs=self.conv_l3_out,
                                                    filters=32, kernel_size=3,
                                                    strides=2, activation=tf.nn.relu)
                # Flatten
                self.flattened_conv_output = tf.layers.flatten(
                    name="conv_output_flattener", inputs=self.conv_l4_out)

                # Hidden Layer
                self.hidden_l_out = tf.layers.dense(name="dense_layer",
                                                    inputs=self.flattened_conv_output,
                                                    units=256, activation=tf.nn.relu)

                # State-Action-Value Distributions (as a flattened vector)
                self.flattened_q_dist = tf.layers.dense(
                    name="flattened_action_value_dist_logits", inputs=self.hidden_l_out,
                    units=num_actions * 51)

                # Unflatten
                self.q_dist_logits = tf.reshape(self.flattened_q_dist,
                                                [-1, num_actions, 51],
                                                name="reshape_q_dist_logits")

                # Softmax State-Action-Value Distributions (per action)
                self.q_dist = tf.nn.softmax(self.q_dist_logits,
                                            name="action_value_dist", axis=-1)

                # Multiply bin probabilities by value
                self.post_mul = tf.multiply(tf.reshape(tf.range(start=-10, limit=10 + 20/50,
                                            delta=(20/50)), [1, 1, 51]), self.q_dist)

                # Take sum to get the expected state-action values for each action
                self.actions = tf.reduce_sum(self.post_mul, axis=2)

                self.q_value_train_net_max_trn = tf.summary.scalar("q_training_net_max_trn", tf.reduce_max(self.actions))
                self.q_value_train_net_mean_trn = tf.summary.scalar("q_training_net_mean_trn", tf.reduce_mean(self.actions))
                self.q_value_train_net_min_trn = tf.summary.scalar("q_training_net_min_trn", tf.reduce_min(self.actions))

                self.q_value_train_net_max_targ = tf.summary.scalar("q_training_net_max_targ", tf.reduce_max(self.actions))
                self.q_value_train_net_mean_targ = tf.summary.scalar("q_training_net_mean_targ", tf.reduce_mean(self.actions))
                self.q_value_train_net_min_targ = tf.summary.scalar("q_training_net_min_targ", tf.reduce_min(self.actions))

            if not train_net:
                # Find argmax action given expected state-action values at next state
                self.argmax_action = tf.argmax(self.actions, axis=-1,
                                               output_type=tf.int32)

                # Get it's corresponding distribution (this is the target distribution)
                self.argmax_action_distribution = tf.gather_nd(self.q_dist,
                    tf.stack((tf.range(start=0, limit=tf.shape(self.argmax_action)[0]), self.argmax_action),
                             axis=1))  # Axis = 1 => [N, 2]

                self.mean_argmax_next_state_value = tf.summary.scalar("mean_argmax_q_target", tf.reduce_mean(tf.range(start=-10, limit=10 + 20/50,
                        delta=(20/50)) * self.argmax_action_distribution))

                # Placeholder for reward
                self.r = tf.placeholder(name="reward", dtype=tf.float32, shape=(None,))
                self.t = tf.placeholder(name="terminal", dtype=tf.uint8, shape=(None,))

                # Compute Tz (Bellman Operator) on atom of expected state-action-value
                # r + gamma * z clipped to [V_min, V_max]
                self.Tz = tf.clip_by_value(tf.reshape(self.r, [-1, 1]) + 0.99 *
                    tf.cast(tf.reshape(self.t, [-1, 1]), tf.float32) *
                    tf.range(start=-10, limit=10 + 20/50, delta=(20/50)),
                    clip_value_min=-10, clip_value_max=10)

                # Compute bin number (will be floating point).
                self.b = (self.Tz - (-10))/(20/50)

                # Lower and Upper Bins.
                self.l = tf.floor(self.b)
                self.u = tf.ceil(self.b)

                # Add weight to the lower bin based on distance from upper bin to
                # approximate bin index b. (0--b--1. If b = 0.3. Then, assign bin
                # 0, p(b) * 0.7 weight and bin 1, p(Z = z_b) * 0.3 weight.)
                self.indexable_l = tf.stack((tf.reshape(tf.range(start=0, limit=
                    tf.shape(self.argmax_action)[0]), [-1, 1]) * tf.ones((1, 51), dtype=tf.int32),
                    tf.cast(self.l, dtype=tf.int32)), axis=-1)
                self.m_l_vals = self.argmax_action_distribution * (self.u - self.b)
                self.m_l = tf.scatter_nd(self.indexable_l, self.m_l_vals, tf.shape(self.l))

                # Add weight to the lower bin based on distance from upper bin to
                # approximate bin index b.
                self.indexable_u = tf.stack((tf.reshape(tf.range(start=0, limit=
                    tf.shape(self.argmax_action)[0]), [-1, 1]) * tf.ones((1, 51), dtype=tf.int32),
                    tf.cast(self.u, dtype=tf.int32)), axis=-1)
                self.m_u_vals = self.argmax_action_distribution * (self.b - self.l)
                self.m_u = tf.scatter_nd(self.indexable_u, self.m_u_vals, tf.shape(self.u))

                # Add Contributions of both upper and lower parts and
                # stop gradient to not update the target network.
                self.m = tf.stop_gradient(tf.squeeze(self.m_l + self.m_u))

                self.weighted_m = tf.clip_by_value(self.m *
                                 tf.range(start=-10, limit=10 + 20/50, delta=(20/50)),
                                 clip_value_min=-10, clip_value_max=10)

                self.weighted_m_mean = tf.summary.scalar("mean_q_target",
                     tf.reduce_mean(self.weighted_m))

                self.targ_dist = tf.summary.histogram("target_distribution",
                                     self.weighted_m)

                self.targn_summary = tf.summary.merge([self.targ_dist, self.weighted_m_mean, self.q_value_train_net_max_targ,
                                                                            self.q_value_train_net_min_targ,
                                                                            self.q_value_train_net_mean_targ,
                                                       self.mean_argmax_next_state_value])
            else:
                # Given you took this action.
                self.action_placeholder = tf.placeholder(name="action", dtype=tf.int32, shape=[None, ])

                # Compute Q-Dist. for the action.
                self.action_q_dist = tf.gather_nd(self.q_dist,
                                                  tf.stack((tf.range(start=0, limit=
                                                  tf.shape(self.action_placeholder)[0]),
                                                            self.action_placeholder), axis=1))

                self.weighted_q_dist = tf.clip_by_value(self.action_q_dist *
                                 tf.range(start=-10, limit=10 + 20/50, delta=(20/50)),
                                 clip_value_min=-10, clip_value_max=10)

                tnd_summary = tf.summary.histogram("training_net_distribution",
                                                   self.weighted_q_dist)

                tnd_mean_summary = tf.summary.scalar("training_net_distribution_mean", tf.reduce_mean(self.weighted_q_dist))

                # Get target distribution.
                self.m_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 51), name="m_placeholder")
                self.loss_sum = -tf.reduce_sum(self.m_placeholder *
                                           tf.log(self.action_q_dist), axis=-1)

                self.loss = tf.reduce_mean(self.loss_sum)

                l_summary = tf.summary.scalar("loss", self.loss)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
                gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
                grad_norm_summary = tf.summary.histogram("grad_norm", tf.global_norm(gradients))
                gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
                self.train_step = self.optimizer.apply_gradients(zip(gradients, variables))
                self.trnn_summary = tf.summary.merge([tnd_mean_summary, tnd_summary, l_summary, grad_norm_summary, self.q_value_train_net_max_trn,
                                                                            self.q_value_train_net_min_trn,
                                                                            self.q_value_train_net_mean_trn])

    def __init__(self, num_actions, exp_replay_size=100000, eps_start=1, eps_end=0.1,
                 eps_period=500000):
        self.num_actions = num_actions
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.experience_replay = deque(maxlen=exp_replay_size)
        with tf.variable_scope("train_net"):
            self.train_net = self.Model(self.sess, self.num_actions, train_net=True)
        with tf.variable_scope("target_net"):
            self.target_net = self.Model(self.sess, self.num_actions, train_net=False)
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("TensorBoardDir")
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.steps = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_period = eps_period

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

        self.saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)
        # self.profiler = tf.profiler.Profiler(self.sess.graph)

        self.beholder = Beholder("./TensorBoardDir")

    def act(self, x):
        self.steps += 1
        if len(self.experience_replay) < 50000 or np.random.random() < \
                self.eps_start - (self.steps/self.eps_period) * (1 - self.eps_end):
            return np.random.randint(0, self.num_actions)
        else:
            actions = self.sess.run(fetches=self.train_net.actions,
                                    feed_dict={self.train_net.x: x})
            return np.argmax(actions)

    def update(self, x, a, r, x_p, t):
        self.experience_replay.appendleft([x, a, r, x_p, not t])

        if len(self.experience_replay) < 50000 or self.steps % 4 != 0:
            return

        total_loss = 0
        for _ in range(1):
            batch_data = random.sample(self.experience_replay, 32)
            batch_x = np.array([i[0] for i in batch_data])
            batch_a = [i[1] for i in batch_data]
            batch_x_p = np.array([np.array(np.dstack((i[0][:, :, 1:], np.maximum(i[3], i[0][:, :, 3]))))
                                 for i in batch_data])
            batch_r = [i[2] for i in batch_data]
            batch_t = [i[4] for i in batch_data]
            # if self.steps % 1000 == 0:
            #     run_meta = tf.RunMetadata()
            #     m = self.sess.run(self.target_net.m, feed_dict={self.target_net.x: batch_x_p,
            #                                                 self.target_net.r: batch_r},
            #                       options=tf.RunOptions(
            #                           trace_level=tf.RunOptions.FULL_TRACE),
            #                       run_metadata=run_meta)
            #
            #     loss, _ = self.sess.run([self.train_net.loss, self.train_net.train_step], feed_dict={self.train_net.x: batch_x,
            #                                                 self.train_net.action_placeholder: batch_a,
            #                                                       self.train_net.m_placeholder: m},
            #                             options=tf.RunOptions(
            #                                 trace_level=tf.RunOptions.FULL_TRACE),
            #                             run_metadata=run_meta)
            #     self.profiler.add_step(self.steps, run_meta)
            #
            #     # Profile the parameters of your model.
            #     self.profiler.profile_name_scope(options=(tf.profiler.ProfileOptionBuilder
            #                                          .trainable_variables_parameter()))
            #
            #     # Or profile the timing of your model operations.
            #     opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
            #     self.profiler.profile_operations(options=opts)
            #
            #     # Or you can generate a timeline:
            #     opts = (tf.profiler.ProfileOptionBuilder(
            #         tf.profiler.ProfileOptionBuilder.time_and_memory())
            #             .with_step(i)
            #             .with_timeline_output("timeline-at-" + str(i)).build())
            #     self.profiler.profile_graph(options=opts)
            #
            # else:
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

            self.writer.add_summary(targn_summary, self.steps)
            self.writer.add_summary(trnn_summary, self.steps)

            total_loss += loss

            self.beholder.update(self.sess, frame=batch_x[0], arrays=[m, Tz, b, u, l, indexable_u, indexable_l, m_u_vals, m_l_vals, m_u, m_l])

        if self.steps > 0 and self.steps % 10000 == 0:
            self.sess.run(self.copy_operation)
            print("Copied to target. Current Loss: ", total_loss)

        if self.steps > 0 and self.steps % 50000 == 0:
            self.saver.save(self.sess, "Models/model",
                            global_step=self.steps,
                            write_meta_graph=(self.steps <= 50000))


def test_agent_forward():
    c = C51Agent(4)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        #print(sess.run((c.train_net.m, c.train_net.actions, c.train_net.post_mul[0, 0, :], c.train_net.q_dist),
        #                feed_dict={c.train_net.x: np.ones([2, 84, 84, 4]),
        #                           c.train_net.r: np.array([1, 1])})[0])

        c.update(np.ones([2, 84, 84, 4]), [1, 1], [1, 1], np.ones([2, 84, 84, 4]), 1)


# test_agent_forward()

# TODO: Implement act() and update() in C51Agent(). Need to ensure two copies
# TODO: are made (train and target)

# TODO: In another 2 files, implement utility functions (pre-process,
# TODO: frame history buffer-handling) and experience replay functions.

