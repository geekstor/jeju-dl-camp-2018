import numpy as np
import tensorflow as tf
from configuration import ConfigurationManager

'''
File: agent.py

Usage:
    Agent Class: Generic wrapper for agent.
    Extensions: Distributional Agent. Implements distribution method.

'''

# Contains **Abstract** Agents. All functions are pure virtual
# (will error if any function is not implemented)


class Agent:
    def __init__(self):
        self.predict_calls = 0
        self.evaluate_calls = 0
        self.train_calls = 0

    # Uses exploration strategy/risk-sensitive strategies and returns
    # non-greedy actions appropriately (at least with respect to the true risk-neutral
    # expected value.) Will likely make use of this class' greedy_action method.
    def predict(self, x):
        self.predict_calls += 1

    def evaluate(self, x):
        self.evaluate_calls += 1

    def train(self, x, a, r, x_p, t):
        self.train_calls += 1

    def y(self, x):
        raise NotImplementedError

    def num_actions(self):
        raise NotImplementedError


class BaseDQNBasedAgent(Agent):
    required_params = ["MINIBATCH_SIZE", "COPY_TARGET_FREQ"]

    def __init__(self, cfg_parser: ConfigurationManager, head):
        super().__init__()

        from util.util import build_train_and_target_general_network_with_head, get_session
        self.sess = get_session(cfg_parser)

        self.train_network_base, self.train_network, \
            self.target_network_base, self.target_network, self.copy_op, self.saver = \
                build_train_and_target_general_network_with_head(head, cfg_parser)

        from memory.experience_replay import ExperienceReplay
        self.experience_replay = ExperienceReplay(cfg_parser)

        self.cfg_parser = cfg_parser

        from function_approximator.head import QNetworkHead

        self.train_network: QNetworkHead

        self.target_network: QNetworkHead

        cfg_parser["NUM_ACTIONS"] = self.train_network.num_actions

        self.cfg = cfg_parser.parse_and_return_dictionary(
            "AGENT", BaseDQNBasedAgent.required_params)

        self.train_step = None

        self.action_placeholder = tf.placeholder(name="action",
                                                 dtype=tf.int32, shape=[None, ])
        self.reward_placeholder = tf.placeholder(name="reward", dtype=tf.float32,
                                                 shape=(None,))
        # TODO: Optimize memory uint8 -> bool (check if casting works to float)
        self.terminal_placeholder = tf.placeholder(name="terminal", dtype=tf.uint8,
                                                   shape=(None,))

        self.predict_calls = 0
        self.train_calls = 0
        self.num_updates = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)

        self.batch_dim_range = tf.range(tf.shape(self.train_network_base.x)[0],
                                   dtype=tf.int32)

        self.policy = None

    def prepare(self, loss_op):
        from optimizer.optimizer import get_optimizer
        self.train_step = get_optimizer(self.cfg_parser, loss_op,
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='train_net'),
                                        global_step=self.num_updates)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(self.copy_op)

        from action_policy.action_policy import Policy
        self.policy = Policy(self.cfg_parser, self)

    def bellman_op(self, distribution):
        r = self.reward_placeholder[:, tf.newaxis]
        casted_t = tf.cast(self.terminal_placeholder, dtype=tf.float32)
        t = casted_t[:, tf.newaxis]
        return r + self.cfg["DISCOUNT_FACTOR"] * t * distribution

    def batch_experiences(self, experiences):
        batch_x = np.array([i[0] for i in experiences])
        batch_a = [i[1] for i in experiences]
        batch_x_p = np.array([i[3] for i in experiences])
        batch_r = [i[2] for i in experiences]
        batch_t = [i[4] for i in experiences]

        feed_dict = {self.train_network_base.x: batch_x,
                     self.action_placeholder: batch_a,
                     self.reward_placeholder: batch_r,
                     self.target_network_base.x: batch_x_p,
                     self.terminal_placeholder: batch_t}

        return feed_dict

    def learn(self, experiences):
        feed_dict = self.batch_experiences(experiences)
        return self.sess.run(fetches=self.train_step, feed_dict=feed_dict)

    def y(self, x):
        raise NotImplementedError

    def act(self, x):
        raise NotImplementedError

    def predict(self, x):
        super().predict(x)
        return self.act(x)

    def evaluate(self, x):
        super().evaluate(x)
        raise NotImplementedError

    def add(self, x, a, r, x_p, t):
        self.experience_replay.add([x, a, r, x_p, not t])

    def train(self, x, a, r, x_p, t):
        super().train(x, a, r, x_p, t)
        assert(self.predict_calls == self.train_calls)
        self.add(x, a, r, x_p, t)

        if self.train_calls % self.cfg["UPDATE_FREQUENCY"] != 0:
            return

        if self.experience_replay.size() > self.cfg["MINIBATCH_SIZE"]:
            self.learn(self.experience_replay.sample(self.cfg["MINIBATCH_SIZE"]))

            global_step = self.num_updates.eval(self.sess)
            if global_step % self.cfg["COPY_TARGET_FREQ"] == 0:
                self.copy()

            if global_step % self.cfg["SAVE_FREQ"] == 0:
                self.save_agent()

    def copy(self):
        self.sess.run(self.copy_op)
        global_step = self.num_updates.eval(self.sess)
        print("Copied to target! Completed ", global_step, " updates!")

    def save_agent(self):
        global_step = self.num_updates.eval(self.sess)
        print("Saving agent. Completed ", global_step, " updates!")
        self.saver: tf.train.Saver
        self.saver.save(sess=self.sess, save_path=self.cfg_parser["TRAIN_FOLDER"] +
                        "/model.ckpt", global_step=self.num_updates,
                        write_meta_graph=True)

    def num_actions(self):
        return self.train_network.num_actions


class DistributionalAgent(BaseDQNBasedAgent):
    def __init__(self, cfg_parser, head):
        super().__init__(cfg_parser, head)

        from function_approximator.head import DistributionalHead
        self.train_network: DistributionalHead
        self.target_network: DistributionalHead

        flat_indices_chosen_actions = tf.stack([self.batch_dim_range,
                                                self.action_placeholder],
                                               axis=1)
        self.dist_of_chosen_actions = tf.gather_nd(self.train_network.q_dist,
                                                   flat_indices_chosen_actions)

        # OPS. for computing target quantile dist.
        flat_indices_for_argmax_action_target_net = tf.stack([
            self.batch_dim_range, self.target_network.greedy_action], axis=1)
        self.dist_of_target_max_q = \
            tf.gather_nd(self.target_network.q_dist,
                         flat_indices_for_argmax_action_target_net)

    def act(self, x):
        return self.policy.act([x])

    def y(self, x):
        return self.sess.run(self.train_network.q,
                             feed_dict={self.train_network_base.x: x})

