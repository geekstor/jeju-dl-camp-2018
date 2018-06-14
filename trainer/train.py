import os
import shutil
import time

import cv2
import gym
import numpy as np
import tensorflow as tf

import params
from agent import C51Agent


def preprocess(x):
    return cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), tuple(params.STATE_DIMENSIONS),
                      interpolation=cv2.INTER_LINEAR)


class Manager:
    def __init__(self):
        if params.GLOBAL_MANAGER is None:
            params.GLOBAL_MANAGER = self
        else:
            assert("Already created Manager. Meant to be used as a singleton. "
                   "Use of multiple managers not supported yet.")
        self.env = gym.make(params.GYM_ENV_NAME)
        self.agent = None
        self.hist_buffer = None
        self.timestep = 0
        self.num_updates = 0
        self.num_episodes = 0
        self.avg_ep_reward = 0
        self.avg_ep_length = 0
        if params.ACTIONS_SPECIFICATION is None:
            self.actions = list(range(self.env.action_space.n))
        else:
            self.actions = params.ACTIONS_SPECIFICATION

    def reset(self):
        self.hist_buffer = np.zeros(params.STATE_DIMENSIONS + [params.HISTORY_LEN],
                                    dtype=np.uint8)
        return preprocess(self.env.reset())

    def act(self, a):
        done = False
        act_rep = 0
        x_primes = []
        act_rep_reward = 0

        while not done and act_rep < params.ACTION_REPEAT:
            x_prime, reward, done, _ = self.env.step(self.actions[a])
            x_primes.append(x_prime)
            act_rep += 1
            act_rep_reward += reward
        x_prime = np.round(np.average(x_primes, axis=0)).astype(
            dtype=np.uint8)

        reward = act_rep_reward
        x_prime = preprocess(x_prime)
        return self.hist_buffer, a, reward, x_prime, done

    def train(self, agent):
        self.agent = agent
        print("Filling replay buffer.")
        x = self.reset()
        for counter in range(params.REPLAY_START_SIZE):
            self.hist_buffer = np.roll(self.hist_buffer, shift=-1, axis=2)
            self.hist_buffer[:, :, -1] = x

            a = np.random.randint(0, len(self.actions))
            x, a, r, x_prime, done = self.act(a)
            self.agent.add(x, a, r, x_prime, done)
            x = x_prime

            if done:
                print("Finished Episode. Completed adding " + str(counter) +
                      " of " + str(params.REPLAY_START_SIZE) + " transitions to buffer.")
                x = self.reset()
        del self.env
        print("Begin training.")
        self.env = gym.make(params.GYM_ENV_NAME)
        self.env = gym.wrappers.Monitor(self.env, params.VIDEOS_FOLDER, video_callable=
            lambda e_id: e_id % params.EPISODE_RECORD_FREQ == 0)
        x = self.reset()

        ep_length = 0
        ep_reward = 0
        start_time = time.time()
        learning_start_time = start_time

        while self.timestep < params.MAX_TIMESTEPS:
            self.hist_buffer = np.roll(self.hist_buffer, shift=-1, axis=2)
            self.hist_buffer[:, :, -1] = x

            a = self.agent.act([self.hist_buffer])
            x, a, r, x_prime, done = self.act(a)

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

                x = self.reset()


    # ion()
    # imshow(np.concatenate([hist_buffer[:, :, 0], hist_buffer[:, :, 1],
    #                        hist_buffer[:, :, 2], hist_buffer[:, :, 2]]),
    #        cmap="gray")
    # pause(0.001)


# if params.VIDEOS_FOLDER in os.listdir("."):
#     shutil.rmtree(os.getcwd() + "\\" + params.VIDEOS_FOLDER)
# if params.MODELS_FOLDER in os.listdir("."):
#     shutil.rmtree(os.getcwd() + "\\" + params.MODELS_FOLDER)
# if params.TENSORBOARD_FOLDER in os.listdir("."):
#     shutil.rmtree(os.getcwd() + "\\" + params.TENSORBOARD_FOLDER)

m = Manager()
agent = C51Agent()
m.train(agent)
#show()