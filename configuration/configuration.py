import json

'''
File: configuration.py

Usage:
    ConfigurationManager - Parses static configuration file. File must be a json file.

JSON Input Schema:
    MANAGER : Training Parameters (# Steps, Episodes, etc.)
    HEAD : nb_atoms 
    NETWORK : Architecture of Network
    OPTIMIZER: Optimizer Hyperparameters (Type, LR, Epsilon)
    AGENT : RL Agent Configuration (Type, Minibatch Size, etc.)
    EXPERIENCE_REPLAY : Experience Replay Size
    ENVIRONMENT : Environment Parameters (Type, Env Name, Level Name, Episodic Life, History Len, Scaled Float, Clip Rewards)

'''

class ConfigurationManager:
    def __init__(self, config_file_loc):
        file = open(config_file_loc)
        self.parsed_json = json.load(file)

    def parse_and_return_dictionary(self, section_name,
                                    required_params):

        assert(section_name in self.parsed_json)
        this_section = self.parsed_json[section_name]

        assert(param in this_section
                for param in required_params)

        return this_section

    def __getitem__(self, key):


        return self.parsed_json[key]

    def __setitem__(self, key, value):
        self.parsed_json[key] = value
