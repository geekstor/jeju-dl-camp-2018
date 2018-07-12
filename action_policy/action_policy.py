import numpy as np
import tensorflow as tf

from configuration import ConfigurationManager

'''
File: action_policy.py

Usage:
    Implementations of multiple exploration strategies (Epsilon Greedy, Softmax)

'''

# TODO: Get greedy action (TF) op. and return appropriate action (TF) op. Instead of
# running things in numpy/CPU.

# TODO: Use Policy!


class Policy:
    required_params = ["POLICY_TYPE", "EXPECTATION_FUNCTION"]

    def __init__(self, cfg_parser: ConfigurationManager, pre_action_op):
        act_plcy_cfg = cfg_parser.parse_and_return_dictionary(
            "POLICY", Policy.required_params)

        if act_plcy_cfg["EXPECTATION_FUNCTION"] == "IDENTITY":
            # TODO: Assumes distributional perspective where last axis corresponds
            # TODO: to atoms.
            self.expectation = tf.reduce_mean(pre_action_op, axis=-1)

        if act_plcy_cfg["POLICY_TYPE"] == "EPSILON_GREEDY":
            self.policy = EpsilonGreedy(cfg_parser, self.expectation)
        else:
            raise NotImplementedError

    def act(self, x):
        self.policy.act(x)


class EpsilonGreedy:
    required_params = ["EPSILON_START", "EPSILON_END", "EPSILON_FINAL_TIMESTEP"]

    def __init__(self, cfg_parser, expectation):
        xpl_policy_cfg = cfg_parser.parse_and_return_dictionary(
            "POLICY", EpsilonGreedy.required_params,
            keep_section=True)

        self.EPSILON_START = xpl_policy_cfg["EPSILON_START"]
        self.EPSILON_END = xpl_policy_cfg["EPSILON_END"]
        self.EPSILON_FINAL_STEP = xpl_policy_cfg["EPSILON_FINAL_TIMESTEP"]

        self.greedy_action = tf.argmax(expectation, axis=-1)

    def get_eps(self):
        global timestep
        return self.EPSILON_START - \
        (min(self.EPSILON_FINAL_STEP, timestep) /
         self.EPSILON_FINAL_STEP) * \
        (1 - self.EPSILON_END)

    def act(self, x):
        global num_actions
        if np.random.random() < self.get_eps():
            return np.random.randint(0, num_actions)
        else:
            return
