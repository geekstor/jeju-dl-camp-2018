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
        f = config_file_loc.open()
        self.parsed_json = json.load(f)

    def parse_and_return_dictionary(self, section_name,
                                    required_params):
        this_section = []

        if "." in section_name:
            return self[section_name]

        if section_name in self.parsed_json:
            this_section = self.parsed_json[section_name]

            assert(param in this_section for param in required_params)

        return this_section

    def __getitem__(self, key):
        keys = key.split('.')
        rv = self.parsed_json
        for k in keys:
            rv = rv[k]
        return rv

    def __setitem__(self, key, value):
        keys = key.split('.')
        rv = self.parsed_json
        for i, k in zip(range(len(keys)), keys):
            if i == len(keys) - 1:
                break
            rv = rv[k]
        rv[key] = value
