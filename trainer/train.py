import os
import sys
import numpy as np

'''
File: train.py

Usage:
    Receives Configuration File. Propagates to Agent, Memory, Optimizer, Function_Approximator, Environment.

'''

if len(sys.argv) < 2:
    assert "Configuration File Required."

from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
config_file_loc = parent_dir / "configuration" / sys.argv[1]

from configuration import ConfigurationManager
cfg_manager = ConfigurationManager(config_file_loc)

import time
start_time = time.time()

os.makedirs(parent_dir / str(start_time))

cfg_manager["TRAIN_FOLDER"] = str(parent_dir / str(start_time))

from shutil import copyfile
copyfile(config_file_loc, parent_dir / str(start_time) / "config.json")

from environment import Environment
e = Environment(cfg_manager)

# Get Defaults for Network Configuration. Defined in global scope!
cfg_manager["DEFAULT_NUM_ACTIONS"] = e.num_actions()
cfg_manager["DEFAULT_OBS_DIMS"] = e.observation_dims()
print(e.num_actions())

# Parse Agent. Also parses Optimizer, Network, and (TODO: Expl. Pol.)
agent = None
if cfg_manager["AGENT"]["TYPE"] == "CATEGORICAL":
    from agent import categorical_agent
    agent = categorical_agent.CategoricalAgent(cfg_manager)
elif cfg_manager["AGENT"]["TYPE"] == "QUANTILE_REGRESSION":
    from agent import quantile_regression
    agent = quantile_regression.QuantileRegressionAgent(cfg_manager)

if "AVERAGE_REWARD_WINDOW" not in cfg_manager["MANAGER"]:
    cfg_manager["MANAGER"]["AVERAGE_REWARD_WINDOW"] = 0
window_start_bound = -cfg_manager["MANAGER"]["AVERAGE_REWARD_WINDOW"]

"""
Manager Class:

Methods:
    __init__: env, agent, render_buffer (recording episodes)
    run: training loop for the reinforcement learning algorithm

"""
class Manager():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        # Used for recording episodes.
        self.render_buffer = []

    def run(self):
        ep_num = 1 # Number of total episodes.
        ep_steps = 0 # Number of steps per episodes.
        ep_r = 0 # ??
        x = self.env.reset()
        #print(x.shape)
        total_r = [] # ??
        import datetime
        ep_start_time = datetime.datetime.now()
        for step in range(cfg_manager["MANAGER"]["NUM_TRAIN_STEPS"]):
            if "EPISODE_RECORD_FREQ" in cfg_manager["MANAGER"] and \
                    ep_num % cfg_manager["MANAGER"]["EPISODE_RECORD_FREQ"] == 0:
                self.env.render()
                self.render_buffer.append(
                    agent.viz([x], self.env.render(mode="rgb_array"))
                )

            #self.env.render()
            a = agent.act(x)
            x_prime, r, done, _ = self.env.step(a)
            #print("action : ", a)
            #print("reward : ", r)
            #print("cum reward : ", ep_r)
            agent.update(x, a, r, x_prime, done)
            #print("updated agent")
            ep_steps += 1
            ep_r += r
            x = x_prime

            if ep_steps % 100 == 0:
                duration = datetime.datetime.now() - ep_start_time             
                seconds = duration.total_seconds()
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                seconds = seconds % 60
                print("Ep. Running, Steps:", ep_steps, "Time since start:", 
                      "%dh:%dm:%ds" % (hours, minutes, seconds))

            if done:
                total_r.append(ep_r)
                print("Episode Num:.", ep_num,
                      "Steps:", ep_steps,
                      "Episode Reward: ", ep_r,
                      "Mean Reward: ", np.mean(total_r[window_start_bound:]) if
                      len(total_r) > abs(window_start_bound) else "Not Yet Enough Ep.")

                if len(self.render_buffer) > 0:
                    from moviepy.editor import ImageSequenceClip
                    clip = ImageSequenceClip(self.render_buffer, fps=5)
                    clip.write_gif(cfg_manager["TRAIN_FOLDER"] + '/ep' + str(ep_num) + '.gif', fps=5)
                    self.render_buffer = []

                x = self.env.reset()
                ep_num += 1
                ep_steps = 0
                ep_r = 0

m = Manager(e, agent)
m.run()
