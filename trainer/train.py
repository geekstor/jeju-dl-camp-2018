import sys

if len(sys.argv) < 2:
    assert "Configuration File Required."

config_file_loc = "../configuration/" + sys.argv[1]

from configuration import ConfigurationManager
cfg_manager = ConfigurationManager(config_file_loc)

from environment import Environment

e = Environment(cfg_manager)

# Get Defaults for Network Configuration. Defined in global scope!
cfg_manager.parsed_json["DEFAULT_NUM_ACTIONS"] = e.num_actions()
cfg_manager.parsed_json["DEFAULT_OBS_DIMS"] = e.observation_dims()
# Also parses Optimizer, Network, and Expl. Pol.
from agent import quantile_regression
distagent = quantile_regression.QuantileRegressionAgent(cfg_manager)


class Manager():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

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
                #if ep_id % 20 == 0 and ep_id > 0:
                #    distagent.viz_dist([x])

                a = distagent.act(x)[0]
                x_prime, r, done, _ = self.env.step(a)

                distagent.update(x, a, r, x_prime, done)
                in_ep_r += r
                steps += 1

                x = x_prime

            total_r.append(in_ep_r)
            import numpy as np
            print(ep_id, steps, total_r[-1], np.mean(total_r))
            ep_id += 1

m = Manager(e, distagent)
m.run()
#agent = QRAgent() # TODO: Start with a base Agent class and
                  # TODO: inherit for all agents. Set Agent
                  # TODO: in config.
#m.train(agent)
#show()
