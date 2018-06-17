import os
import shutil
import time

import gym
import numpy as np
import tensorflow as tf

import params
import wrappers
from agent import C51Agent


class Manager:
    def __init__(self):
        if params.GLOBAL_MANAGER is None:
            params.GLOBAL_MANAGER = self
        else:
            assert("Already created Manager. Meant to be used as a singleton. "
                   "Use of multiple managers not supported yet.")
        self.env = wrappers.wrap_env(wrappers.make_atari(params.GYM_ENV_NAME))
        self.agent = None
        self.timestep = 0
        self.num_updates = 0
        self.num_episodes = 0
        self.avg_ep_reward = 0
        self.avg_ep_length = 0
        if params.ACTIONS_SPECIFICATION is None:
            self.actions = list(range(self.env.action_space.n))
        else:
            self.actions = params.ACTIONS_SPECIFICATION

    def train(self, agent):
        self.agent = agent
        x = self.env.reset()
        print("Filling replay buffer.")
        for counter in range(params.REPLAY_START_SIZE):
            a = np.random.randint(0, len(self.actions))
            x_prime, r, done, _ = self.env.step(self.actions[a])
            self.agent.add(x, a, r, x_prime, done)
            x = x_prime

            if done:
                print("Finished Episode. Completed adding " + str(counter) +
                      " of " + str(params.REPLAY_START_SIZE) + " transitions to buffer.")
                x = self.env.reset()
                
        del self.env
        print("Begin training.")
        self.env = wrappers.wrap_env(wrappers.make_atari(params.GYM_ENV_NAME))
        self.env = gym.wrappers.Monitor(self.env, params.VIDEOS_FOLDER, video_callable=
            lambda e_id: e_id % params.EPISODE_RECORD_FREQ == 0)
        x = self.env.reset()

        ep_length = 0
        ep_reward = 0
        start_time = time.time()
        learning_start_time = start_time

        while self.timestep < params.MAX_TIMESTEPS:
            a = self.agent.act(np.array([x]))
            x_prime, r, done, _ = self.env.step(self.actions[a])

            if self.timestep % params.UPDATE_FREQUENCY == 0:
                self.agent.update(x, a, r, x_prime, done)
                self.num_updates += 1
            else:
                self.agent.add(x, a, r, x_prime, done)

            x = x_prime

            self.timestep += 1
            ep_length += 1
            ep_reward += r

            if done:
                time_elapsed = time.time() - start_time
                self.avg_ep_reward = (self.avg_ep_reward * self.num_episodes +
                                      ep_reward) / (self.num_episodes + 1)
                self.avg_ep_length = (self.avg_ep_length * self.num_episodes +
                                      ep_length) / (self.num_episodes + 1)
                self.num_episodes += 1

                agent.writer.add_summary(summary=tf.Summary(value=[
                    tf.Summary.Value(tag="average_ep_reward", simple_value=self.avg_ep_reward),
                ]), global_step=self.num_episodes)

                agent.writer.add_summary(summary=tf.Summary(value=[
                    tf.Summary.Value(tag="episode_reward", simple_value=ep_reward),
                ]), global_step=self.num_episodes)

                agent.writer.add_summary(summary=tf.Summary(value=[
                    tf.Summary.Value(tag="average steps per second", simple_value=
                    self.timestep / (time.time() - learning_start_time)),
                ]), global_step=self.num_episodes)

                agent.writer.add_summary(summary=tf.Summary(value=[
                    tf.Summary.Value(tag="average updates per second", simple_value=
                    self.num_updates / (time.time() - learning_start_time)),
                ]), global_step=self.num_episodes)

                agent.writer.add_summary(summary=tf.Summary(value=[
                    tf.Summary.Value(tag="time per episode", simple_value=
                    time_elapsed),
                ]), global_step=self.num_episodes)

                agent.writer.add_summary(summary=tf.Summary(value=[
                    tf.Summary.Value(tag="ep_length", simple_value=ep_length),
                ]), global_step=self.num_episodes)

                agent.writer.add_summary(summary=tf.Summary(value=[
                    tf.Summary.Value(tag="avg_ep_length", simple_value=self.avg_ep_length),
                ]), global_step=self.num_episodes)

                print("# Ep.: ", self.num_episodes,
                      "Ep. Reward: ", ep_reward,
                      "Ep. Length: ", ep_length,
                      "# Updates: ", self.num_updates,
                      "Avg. Ep. Reward: ", self.avg_ep_reward,
                      "Avg. Ep. Length: ", self.avg_ep_length)

                ep_length = 0
                ep_reward = 0
                start_time = time.time()

                x = self.env.reset()


    # ion()
    # imshow(np.concatenate([hist_buffer[:, :, 0], hist_buffer[:, :, 1],
    #                        hist_buffer[:, :, 2], hist_buffer[:, :, 2]]),
    #        cmap="gray")
    # pause(0.001)


if params.VIDEOS_FOLDER in os.listdir("."):
    shutil.rmtree(os.getcwd() + "\\" + params.VIDEOS_FOLDER)
if params.MODELS_FOLDER in os.listdir("."):
    shutil.rmtree(os.getcwd() + "\\" + params.MODELS_FOLDER)
if params.TENSORBOARD_FOLDER in os.listdir("."):
    shutil.rmtree(os.getcwd() + "\\" + params.TENSORBOARD_FOLDER)

m = Manager()
agent = C51Agent()
m.train(agent)
#show()