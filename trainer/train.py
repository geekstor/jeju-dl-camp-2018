import os
import sys
import numpy as np

if len(sys.argv) < 2:
    assert "Configuration File Required."

from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
config_file_loc = parent_dir / "configuration" / sys.argv[1]

from configuration import ConfigurationManager
cfg_manager = ConfigurationManager(config_file_loc)

from environment import Environment
e = Environment(cfg_manager)

# Get Defaults for Network Configuration. Defined in global scope!
cfg_manager["DEFAULT_NUM_ACTIONS"] = e.num_actions()
cfg_manager["DEFAULT_OBS_DIMS"] = e.observation_dims()

# Parse Agent. Also parses Optimizer, Network, and (TODO: Expl. Pol.)
agent = None
if cfg_manager["AGENT"]["TYPE"] == "CATEGORICAL":
    from agent import categorical_agent
    agent = categorical_agent.CategoricalAgent(cfg_manager)
elif cfg_manager["AGENT"]["TYPE"] == "QUANTILE_REGRESSION":
    from agent import quantile_regression
    agent = quantile_regression.QuantileRegressionAgent(cfg_manager)

import time
start_time = time.time()

os.makedirs(parent_dir / str(start_time))
from shutil import copyfile
copyfile(config_file_loc, parent_dir / str(start_time) / "config.json")

if "AVERAGE_REWARD_WINDOW" not in cfg_manager["MANAGER"]:
    cfg_manager["MANAGER"]["AVERAGE_REWARD_WINDOW"] = 0
window_start_bound = -cfg_manager["MANAGER"]["AVERAGE_REWARD_WINDOW"]

class Manager():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        # Used for recording episodes.
        self.render_buffer = []

    def run(self):
        ep_num = 1
        ep_steps = 0
        ep_r = 0
        x = self.env.reset()
        for step in range(cfg_manager["MANAGER"]["NUM_TRAIN_STEPS"]):
            #if ep_id % cfg_manager["MANAGER"]["EPISODE_RECORD_FREQ"] == 0:
            #    self.render_buffer.append(
            #        agent.viz([x], self.env.render(mode="rgb_array"))
            #    )

            a = agent.act(x)
            x_prime, r, done, _ = self.env.step(a)
            agent.update(x, a, r, x_prime, done)
            ep_r += r
            x = x_prime

            if done:
                total_r.append(ep_r)
                print("Episode Num:.", ep_num,
                      "Steps:", steps,
                      "Episode Reward: ", ep_r,
                      "Mean Reward: ", np.mean(total_r[window_start_bound:]))

                x = self.env.reset()
                ep_num += 1
                ep_steps = 0
                ep_r = 0

            if len(self.render_buffer) > 0:
                from moviepy.editor import ImageSequenceClip
                clip = ImageSequenceClip(self.render_buffer, fps=5)
                clip.write_gif(str(start_time) + '/ep' + str(ep_id) + '.gif', fps=5)
                self.render_buffer = []

#agent = QRAgent() # TODO: Start with a base Agent class and
                  # TODO: inherit for all agents. Set Agent
                  # TODO: in config.
#m.train(agent)
#show()
m = Manager(e, agent)
m.run()
