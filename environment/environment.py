from configuration import ConfigurationManager

'''
File: environment.py

Usage:
    Defines environment for the agent.
    Supports: Gym, Gym-Retro.

Params:
    ENVIRONMENT_TYPE
    GYM_ENV_NAME
    GYM_ENV_LEVEL

Methods:
    step: Next step in the environment.
    reset: Reset to the first frame of the environment.
    render: Graphically render the environment with settings.
    num_actions: returns the current action space length.
    observation_dims: returns the current observation dimensional shape.


'''

class Environment:
    required_params = ["ENVIRONMENT_TYPE"]

    def __init__(self, config_parser: ConfigurationManager):
        env_cfg = config_parser.parse_and_return_dictionary(
            "ENVIRONMENT", Environment.required_params)

        if env_cfg["ENVIRONMENT_TYPE"] == "GYM":
            import gym
            gym_env_required_params = ["GYM_ENV_NAME"]
            gym_env_cfg = config_parser.parse_and_return_dictionary(
                "ENVIRONMENT", gym_env_required_params)
            self.env = gym.make(gym_env_cfg["GYM_ENV_NAME"])
        elif env_cfg["ENVIRONMENT_TYPE"] == "GYM-RETRO":
            import retro
            gym_env_required_params = ["GYM_ENV_NAME"]
            gym_env_cfg = config_parser.parse_and_return_dictionary(
                "ENVIRONMENT", gym_env_required_params)
            self.env = retro.make(game=gym_env_cfg["GYM_ENV_NAME"], state=gym_env_cfg["GYM_ENV_LEVEL"])
<<<<<<< HEAD
            from trainer.wrappers import wrap_env, SonicActionWrapper
            if "WRAP_SONIC" in gym_env_cfg and gym_env_cfg["WRAP_SONIC"]:
                self.env = SonicActionWrapper(self.env)
            wrap_env(self.env, gym_env_cfg)
=======
            from trainer.wrappers import wrap_env
            self.env = wrap_env(self.env, gym_env_cfg)
>>>>>>> c840dbc52f76eef9b75df1e0346e5b959eb47154
        else:
            raise NotImplementedError

    def step(self, a):
        return self.env.step(a)

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        # Add distribution here.
        return self.env.render(mode)

    def num_actions(self):
        # Assumes gym env. Set this var. for other envs.
        return self.env.action_space.n

    def observation_dims(self):
        # Assumes gym env. Set this var. for other envs.
        return self.env.observation_space.shape
