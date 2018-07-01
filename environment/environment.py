from configuration import ConfigurationManager


class Environment:
    required_params = ["ENVIRONMENT_TYPE"]

    def __init__(self, config_parser: ConfigurationManager):
        env_cfg = config_parser.parse_and_return_dictionary(
            "ENVIRONMENT", Environment.required_params, keep_section=True)

        if env_cfg["ENVIRONMENT_TYPE"] == "GYM":
            import gym
            gym_env_required_params = ["GYM_ENV_NAME"]
            gym_env_cfg = config_parser.parse_and_return_dictionary(
                "ENVIRONMENT", gym_env_required_params)
            self.env = gym.make(gym_env_cfg["GYM_ENV_NAME"])
        else:
            raise NotImplementedError

    def step(self, a):
        return self.env.step(a)

    def reset(self):
        return self.env.reset()

    def render(self):
        # Add distribution here.
        return self.env.render()

    def num_actions(self):
        # Assumes gym env. Set this var. for other envs.
        return self.env.action_space.n

    def observation_dims(self):
        # Assumes gym env. Set this var. for other envs.
        return self.env.observation_space.shape

