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
    def __init__(self, cfg_parser: ConfigurationManager):
        pass

    # Uses exploration strategy/risk-sensitive strategies and returns
    # non-greedy actions appropriately (at least with respect to the true risk-neutral
    # expected value.) Will likely make use of this class' greedy_action method.
    def act(self, x):
        raise NotImplementedError

    def update(self, x, a, r, x_p, t):
        raise NotImplementedError

class BaseQValueBasedAgent(Agent):
    required_params = ["MINIBATCH_SIZE", "COPY_TARGET_FREQ"]

    def __init__(self, cfg_parser: ConfigurationManager, head):
        super().__init__(cfg_parser)

        from util.util import build_train_and_target_general_network_with_head, get_session
        self.sess = get_session(cfg_parser)

        self.train_network_base, self.train_network, \
            self.target_network_base, self.target_network, self.copy_op = \
                build_train_and_target_general_network_with_head(head, cfg_parser)

        from memory.experience_replay import ExperienceReplay
        self.experience_replay = ExperienceReplay(cfg_parser)

        self.cfg_parser = cfg_parser

        self.cfg = cfg_parser.parse_and_return_dictionary(
            "AGENT", BaseQValueBasedAgent.required_params)

        self.num_updates = 0

    def prepare(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(self.copy_op)

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
        raise NotImplementedError

    def add(self, x, a, r, x_p, t):
        self.experience_replay.add([x, a, r, x_p, not t])

    def update(self, x, a, r, x_p, t):
        self.num_updates += 1
        self.add(x, a, r, x_p, t)

        if self.experience_replay.size() > self.cfg["MINIBATCH_SIZE"]:
            self.learn(self.experience_replay.sample(self.cfg["MINIBATCH_SIZE"]))

        if self.num_updates > 0 and \
            self.num_updates % self.cfg["COPY_TARGET_FREQ"] == 0:
            self.sess.run(self.copy_op)
            print("Copied.")

class DistributionalAgent(BaseQValueBasedAgent):
    def __init__(self, cfg_parser, head):
        super().__init__(cfg_parser, head)

    # Returns distribution over state-action values.
    def distribution(self, x):
        raise NotImplementedError
