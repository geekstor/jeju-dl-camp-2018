import json


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
