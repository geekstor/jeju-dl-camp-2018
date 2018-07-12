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
    required_params = ["COPY_TARGET_FREQUENCY",
                       "UPDATE_FREQUENCY", "DISCOUNT_FACTOR",
                       "V_MIN", "V_MAX"]

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
            "AGENT", CategoricalAgent.required_params)

        from function_approximator import GeneralNetwork
        with tf.variable_scope("train_net"):
            self.train_network_base = GeneralNetwork(cfg_parser)
            self.train_network = SoftmaxFixedAtomsDistributionalHead(
                cfg_parser, self.train_network_base)
        with tf.variable_scope("target_net"):
            self.target_network_base = GeneralNetwork(cfg_parser)
            self.target_network = SoftmaxFixedAtomsDistributionalHead(
                cfg_parser, self.target_network_base)

        from util.util import get_copy_op, get_vars_with_scope
        self.copy_operation = get_copy_op("train_net",
                                          "target_net")

        from memory.experience_replay import ExperienceReplay
        self.experience_replay = ExperienceReplay(cfg_parser)

        self.build_networks()

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.sess.run(self.copy_operation)

        self.setup_animation()
        self.saver = tf.train.Saver(var_list=get_vars_with_scope("train_net") + get_vars_with_scope("target_net"),
                                    max_to_keep=1000, keep_checkpoint_every_n_hours=1)

    def setup_animation(self):
        num_actions = len(self.train_network.actions)

        plt.subplot2grid((num_actions, num_actions), (0, 0),
                         colspan=num_actions - 1, rowspan=num_actions)

        self.img_obj = plt.imshow(np.zeros(shape=(84, 84, 3)), aspect="auto", interpolation="nearest")

        l, s = np.linspace(self.cfg["V_MIN"],
                           self.cfg["V_MAX"],
                           self.train_network.cfg["NB_ATOMS"],
                           retstep=True)
        self.bar_obj = [None] * num_actions
        for i in range(num_actions):
            plt.subplot2grid((num_actions, num_actions), (i, num_actions - 1),
                             colspan=1, rowspan=1)
            # plt.subplot(len(self.train_network.actions), 2, 2 * (i + 1))
            self.bar_obj[i] = plt.bar(l - s / 2., height=[1 / self.train_network.cfg["NB_ATOMS"]] *
                                       self.train_network.cfg["NB_ATOMS"], width=s,
                    color="brown", edgecolor="red", linewidth=0.5, align="edge")

        plt.gcf().canvas.draw()

    def build_networks(self):
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

                # Take sum to get the expected state-action values for each action
                # graph.actions = tf.reduce_sum(graph.post_mul, axis=2,
                #                               name="expected_state_action_value")

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

        # Placeholder for reward and terminal
        self.r = tf.placeholder(name="reward", dtype=tf.float32, shape=(None,))
        self.t = tf.placeholder(name="terminal", dtype=tf.uint8, shape=(None,))
        # TODO: Optimize memory uint8 -> bool (check if casting works to float)

        self.Tz = tf.clip_by_value(tf.reshape(self.r, [-1, 1]) + self.cfg["DISCOUNT_FACTOR"] *
                                   tf.cast(tf.reshape(self.t, [-1, 1]), tf.float32) * Z,
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

        # Given you took this action.
        self.action_placeholder = tf.placeholder(name="action",
                                                 dtype=tf.int32, shape=[None, ])

        # Compute Q-Dist. for the action.
        self.action_q_dist = tf.gather_nd(self.train_network.y,
                                          tf.stack((batch_size_range,
                                                    self.action_placeholder),
                                                   axis=1))

        self.loss_sum = -tf.reduce_sum(self.m *
                                       tf.log(self.action_q_dist + 1e-5), axis=-1, name="loss")

        self.loss = tf.reduce_mean(self.loss_sum)

        from optimizer.optimizer import get_optimizer
        self.train_step = get_optimizer(self.cfg_parser, self.loss,
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                              scope='train_net'))

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

        return self.sess.run([self.train_step],
                             feed_dict={self.train_network_base.x: batch_x,
                                        self.action_placeholder: batch_a,
                                        self.target_network_base.x: batch_x_p,
                                        self.r: batch_r,
                                        self.t: batch_t})

    def act(self, x):
        if random.random() < 1.0 - (min(10000, self.num_updates) / 10000) * (1 - 0.1):
            return self.train_network.act_to_send(
                random.choice(self.train_network.actions)
            )
        else:
            return self.train_network.act_to_send(self.greedy_action([x])[0])

    def viz(self, x, rgb_x):
        # Plot
        h = np.squeeze(self.sess.run(fetches=self.train_network.y,
                       feed_dict={self.train_network_base.x: x}))

        from scipy.misc import imresize
        self.img_obj.set_data(
            imresize(rgb_x, [rgb_x.shape[0] * 10, rgb_x.shape[1] * 10])
        )
        self.img_obj.autoscale()

        for i in range(h.shape[0]):
            for rect, hi in zip(self.bar_obj[i], h[i]):
                rect.set_height(hi)

        data = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

        return data

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

        if self.num_updates > 0 and "SAVE_FREQ" in self.cfg and \
            self.num_updates % self.cfg["SAVE_FREQ"] == 0:
                self.saver.save(sess=self.sess, global_step=self.num_updates,
                                save_path=self.cfg_parser["TRAIN_FOLDER"] + "/model", write_meta_graph=True)
