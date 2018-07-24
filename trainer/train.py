import os
import sys
import datetime
import numpy as np
from collections import deque

'''
File: train.py

Usage:
    Receives Configuration File. Propagates to Agent, Memory, Optimizer, Function_Approximator, Environment.

'''

if len(sys.argv) < 2:
    raise Exception("Configuration File Required.")

# --- Get Config. File Loc. --- #
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
config_file_loc = parent_dir / "configuration" / sys.argv[1]
# ----------------------------- #

# --- Load Configuration --- #
from configuration import ConfigurationManager
cfg_manager = ConfigurationManager(config_file_loc)
print("Loaded Configuration!")
# -------------------------- #

# --- Create Root Folder --- #
import time
start_time = time.time()
os.makedirs(str(parent_dir / str(start_time)))
cfg_manager["TRAIN_FOLDER"] = str(parent_dir / str(start_time))
print("Created Root Folder: ", str(start_time))
# -------------------------- #

# --- Copy Config. Loc into Root --- #
from shutil import copyfile
copyfile(str(config_file_loc),
         str(parent_dir / str(start_time) / "config.json"))
# ---------------------------------- #

# --- Create Environment --- #
from environment import Environment
e = Environment(cfg_manager)
# -------------------------- #

# --- Load Defaults for Action and Obs. --- #
cfg_manager["DEFAULT_NUM_ACTIONS"] = e.num_actions()
cfg_manager["DEFAULT_OBS_DIMS"] = e.observation_dims()
# ----------------------------------------- #

# --- Wrap environment so that network outputs lead to correct actions --- #
from util.wrappers import NetworkActionToEnvAction
e = NetworkActionToEnvAction(e, cfg_manager)
# ------------------------------------------------------------------------ #

# Parse Agent. Also parses Optimizer, Network, and (TODO: Expl. Pol.)
agent = None
if cfg_manager["AGENT"]["TYPE"] == "CATEGORICAL":
    from agent import categorical_agent
    agent = categorical_agent.CategoricalAgent(cfg_manager)
elif cfg_manager["AGENT"]["TYPE"] == "QUANTILE_REGRESSION":
    from agent import quantile_regression
    agent = quantile_regression.QuantileRegressionAgent(cfg_manager)
elif cfg_manager["AGENT"]["TYPE"] == "IMPLICIT_QUANTILE":
    from agent import iqn
    agent = iqn.ImplicitQuantileAgent(cfg_manager)
else:
    raise Exception("Agent: ", cfg_manager["AGENT"]["TYPE"],
                    " not implemented!")

# --- Set WindowSize (Last N Episode Average Reward) --- #
if "AVERAGE_REWARD_WINDOW" not in cfg_manager["MANAGER"]:
    window_size = None # Default: Average over all episodes.
else: # Average_Reward_Window defined in config.
    window_size = cfg_manager["MANAGER"]["AVERAGE_REWARD_WINDOW"]
# --------------------------------------------------------------- #

"""
Manager Class:

Methods:
    __init__: env, agent
    run: training loop for the reinforcement learning algorithm

"""
class Manager():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self):
        ep_num = 1 # Number of total episodes.
        ep_steps = 0 # Number of steps this episode has lasted for.
        ep_r = 0 # Keeps track of total reward collected in this episode.

        if window_size is None:
            all_r = deque() # Track of all the ep. rewards.
        else:
            all_r = deque(maxlen=window_size) # Track last N ep. rewards.

        ep_start_time = datetime.datetime.now()
        x = self.env.reset()

        for steps in range(cfg_manager["MANAGER"]["NUM_TRAIN_STEPS"]):
            a = self.agent.predict(x)
            x_prime, r, done, _ = self.env.step(a)
            agent.train(x, a, r, x_prime, done)

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
                all_r.append(ep_r)
                print("Episode Num:.", ep_num,
                      "Ep. Steps:", ep_steps,
                      "Total Steps:", steps,
                      "Episode Reward: ", ep_r,
                      "Mean Reward: ", np.mean(all_r) if
                      all_r.maxlen is None or
                      len(all_r) == all_r.maxlen
                      else "Not Yet Enough Ep.")

                x = self.env.reset()
                ep_num += 1
                ep_steps = 0
                ep_r = 0


m = Manager(e, agent)
m.run()

# TODO: Move code below to an evaluate script (after loading agent.)
# # Used for recording episodes.
# self.render_buffer = []

# if "EPISODE_RECORD_FREQ" in cfg_manager["MANAGER"] and \
#         ep_num % cfg_manager["MANAGER"]["EPISODE_RECORD_FREQ"] == 0:
#     self.render_buffer.append(
#         agent.viz([x], self.env.render(mode="rgb_array"))
#     )

# if len(self.render_buffer) > 0:
#     from moviepy.editor import ImageSequenceClip
#     clip = ImageSequenceClip(self.render_buffer, fps=5)
#     clip.write_gif(cfg_manager["TRAIN_FOLDER"] + '/ep' + str(ep_num) + '.gif', fps=5)
#     self.render_buffer = []
