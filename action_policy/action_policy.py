import numpy as np

from configuration import ConfigurationManager


class ActionPolicy:
    required_params = ["ACTION_POLICY_TYPE"]

    def __init__(self, cfg_parser: ConfigurationManager, agent):
        act_plcy_cfg = cfg_parser.parse_and_return_dictionary(
            "ACTION_POLICY", ActionPolicy.required_params,
            keep_section=True)

        if act_plcy_cfg["ACTION_POLICY_TYPE"] == "EPSILON_GREEDY":
            self.action_policy = EpsilonGreedy(cfg_parser, agent)

    def act(self, x):
        self.action_policy.act()


class EpsilonGreedy:
    def __init__(self, cfg_parser, agent):
        self.agent = agent
        self.EPSILON_START = 0
        self.EPSILON_END = 0
        self.EPSILON_FINAL_STEP = 0
        self.num_actions = 0

        assert(hasattr(agent, "num_actions"))
        assert(hasattr(agent, "timestep"))

    def act(self, x, timestep):
        if np.random.random() < self.EPSILON_START - \
                (min(self.EPSILON_FINAL_STEP, timestep) /
                 self.EPSILON_FINAL_STEP) * \
                (1 - self.EPSILON_END):
            return np.random.randint(0, self.num_actions)
        else:
            return self.agent.greedy_action(x)

