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
    def act(self, state):
        raise NotImplementedError

    def greedy_action(self, state):
        raise NotImplementedError

    def learn(self, experiences):
        raise NotImplementedError


class DistributionalAgent(Agent):
    def __init__(self, cfg_parser):
        super().__init__(cfg_parser)

    # Uses exploration strategy/risk-sensitive strategies and returns
    # non-greedy actions appropriately (at least with respect to the true risk-neutral
    # expected value.)
    def act(self, state):
        raise NotImplementedError

    # With Respect to whichever Distorted Distribution we might
    # be using. Returns action.
    def greedy_action(self, state):
        raise NotImplementedError

    def learn(self, experiences):
        raise NotImplementedError

    # Returns distribution over state-action values.
    def distribution(self, state):
        raise NotImplementedError
