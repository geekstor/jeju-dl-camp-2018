import numpy as np
from agent.agent import Agent
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
    required_params = ["POLICY_INPUT", "POLICY_TYPE"]

    def __init__(self, cfg_parser: ConfigurationManager, agent: Agent):
        act_plcy_cfg = cfg_parser.parse_and_return_dictionary(
            "POLICY.EXPLORATION_STRATEGY", Policy.required_params)

        if act_plcy_cfg["TYPE"] == "EPSILON_GREEDY":
            self.policy = EpsilonGreedy(cfg_parser, agent)
        elif act_plcy_cfg["TYPE"] == "SOFTMAX":
            self.policy = SoftMax(agent)
        else:
            raise NotImplementedError

    def act(self, x):
        return self.policy.act(x)


class EpsilonGreedy:
    required_params = ["EPSILON_START", "EPSILON_END", "EPSILON_FINAL_TIMESTEP"]

    def __init__(self, cfg_parser, agent):
        xpl_policy_cfg = cfg_parser.parse_and_return_dictionary(
            "POLICY.EXPLORATION_STRATEGY", EpsilonGreedy.required_params)

        self.EPSILON_START = xpl_policy_cfg["EPSILON_START"]
        self.EPSILON_END = xpl_policy_cfg["EPSILON_END"]
        self.EPSILON_FINAL_STEP = xpl_policy_cfg["EPSILON_FINAL_TIMESTEP"]

        assert(hasattr(agent, "predict_calls"))

        self.agent = agent

    def get_eps(self):
        return self.EPSILON_START - \
        (min(self.EPSILON_FINAL_STEP, self.agent.predict_calls) /
         self.EPSILON_FINAL_STEP) * \
        (1 - self.EPSILON_END)

    def act(self, x):
        if np.random.random() < self.get_eps():
            return np.random.randint(0, self.agent.num_actions())
        else:
            return np.argmax(self.agent.y(x))


class SoftMax:
    required_params = []

    def __init__(self, agent):
        self.agent = agent

    def act(self, x):
        q_or_pi_logits = self.agent.y(x)
        assert(np.sum(q_or_pi_logits) != 1.0)
        return np.random.choice(list(range(len(q_or_pi_logits))),
                                p=np.exp(x - max(x)) / np.sum(np.exp(x - max(x))))

