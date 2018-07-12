import random
from collections import deque

from configuration import ConfigurationManager

'''
File: experience_replay.py

Usage:
    Defines the experience replay for minibatch sampling during agent training.
    Class ExperienceReplay: Wrapper to implement add, sample, size.

'''

class ExperienceReplay:
    required_params = ["EXPERIENCE_REPLAY_SIZE"]

    def __init__(self, config_parser: ConfigurationManager):
        self.cfg = config_parser.parse_and_return_dictionary(
            "EXPERIENCE_REPLAY", ExperienceReplay.required_params)

        self.memory = deque(maxlen=self.cfg["EXPERIENCE_REPLAY_SIZE"])

    # Agent manages experience addition/sampling
    # (Can store data however it wants and sample later.)
    def add(self, experience):
        self.memory.append(experience)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def size(self):
        return len(self.memory)
