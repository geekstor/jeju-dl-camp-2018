import os
import sys

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


class Manager():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.render_buffer = []

    def run(self):
        ep_id = 0
        steps = 0
        from collections import deque
        total_r = deque(maxlen=10)
        while True:
            x = self.env.reset()
            done = False
            in_ep_r = 0
            while not done:
                if ep_id % 20 == 0 and ep_id > 0:
                    self.render_buffer.append(
                        agent.viz_dist([x], self.env.render(mode="rgb_array"))
                    )

                a = agent.act(x)
                x_prime, r, done, _ = self.env.step(a)

                agent.update(x, a, r, x_prime, done)
                in_ep_r += r
                steps += 1

                x = x_prime

            total_r.append(in_ep_r)
            import numpy as np
            print(ep_id, steps, total_r[-1], np.mean(total_r))
            ep_id += 1
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
