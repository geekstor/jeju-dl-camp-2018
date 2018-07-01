import configparser as cgp


class ConfigurationManager:
    def __init__(self, config_file_loc):
        self.config_parser = cgp.ConfigParser()
        self.config_parser.read(config_file_loc)

    def parse_and_return_dictionary(self, section_name,
                                    required_params, keep_section=False):
        sections = self.config_parser.sections()

        if len(sections) == 0:
            print("No Sections Found! "
                  "Please check for multiple calls to this function!")

        assert(section_name in sections)
        this_section = sections[section_name]

        assert([param in this_section
                for param in required_params])

        # Ensure only one call to this function for each section
        # if keep_section is not used.
        if not keep_section:
            self.config_parser.remove_section(section_name)

        if len(sections) == 0:
            print("All Sections Parsed!")

        return sections[section_name]
