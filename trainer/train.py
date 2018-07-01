import importlib
import sys

if len(sys.argv) < 2:
    assert "Configuration File Required."
config_file_loc = importlib.import_module(sys.argv[1])

from configuration import ConfigurationManager
cfg_manager = ConfigurationManager(config_file_loc)
cfg_parser = cfg_manager.config_parser

from environment import Environment
env = Environment(cfg_parser)

# Get Defaults for Network Configuration.
num_actions = env.num_actions()
observation_dims = env.observation_dims()

# Also parses Optimizer, Network, and Expl. Pol.
from agent import Agent
agent = Agent(cfg_parser, num_actions, observation_dims)


m = Manager()
#agent = QRAgent() # TODO: Start with a base Agent class and
                  # TODO: inherit for all agents. Set Agent
                  # TODO: in config.
#m.train(agent)
#show()