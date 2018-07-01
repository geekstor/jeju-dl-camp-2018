from agent import CategoricalAgent
from configuration import ConfigurationManager


class Agent:
    def __init__(self, cfg_parser: ConfigurationManager, *args):
        agent_config = cfg_parser.parse_and_return_dictionary("AGENT",
                                                              ["AGENT_TYPE"],
                                                              keep_section=True)
        if agent_config["AGENT_TYPE"] == "CATEGORICAL":
            self.agent = CategoricalAgent(cfg_parser, *args)
        elif agent_config["AGENT_TYPE"] == "QUANTILE_REGRESSION":
            self.agent = QuantileRegressionAgent(cfg_parser)
        else:
            raise NotImplementedError

    def greedy_action(self, state):
        return self.agent.greedy_action(state)

    def learn(self, experiences):
        return self.agent.learn(experiences)


class DistributionalAgent(Agent):
    def __init__(self, cfg_parser):
        super().__init__(cfg_parser)

    def distribution(self):
        raise NotImplementedError

