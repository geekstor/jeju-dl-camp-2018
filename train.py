import os
import cv2
import gym
import time
import shutil
import numpy as np
import tensorflow as tf
from agent import C51Agent
from collections import deque
from skimage import img_as_ubyte
from skimage.transform import resize, rescale
from skimage.color import rgb2gray
from matplotlib.pyplot import imshow, show, pause, ion, draw

def preprocess(x):
    return cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (42, 42), interpolation=cv2.INTER_AREA)


def train(env, agent, max_timesteps, history_len=4):
    hist_buffer = np.zeros([42, 42, history_len])
    x = preprocess(env.reset())
    total_r = 0

    avg_reward = 0
    num_episodes = 0

    done = False
    timestep = 0
    start_time = time.time()

    ep_length = 0
    avg_ep_length = 0

    # ion()
    # imshow(np.concatenate([hist_buffer[:, :, 0], hist_buffer[:, :, 1],
    #                        hist_buffer[:, :, 2], hist_buffer[:, :, 2]]),
    #        cmap="gray")
    # pause(1)
    while timestep < max_timesteps:
        if done:
            x = preprocess(env.reset())
            hist_buffer = np.zeros([42, 42, history_len], dtype=np.uint8)
            avg_reward = ((num_episodes * avg_reward) + total_r) / (num_episodes + 1)
            avg_ep_length = ((num_episodes * avg_ep_length) + ep_length) / (num_episodes + 1)
            print("Steps Completed: ", agent.steps,
                  "Total Episode Reward: ", total_r,
                  "Average Reward: ", avg_reward, " | ",
                  str(agent.steps / (time.time() - start_time)) +
                  " steps per second.")

            agent.writer.add_summary(summary = tf.Summary(value=[
                tf.Summary.Value(tag="average_reward", simple_value=avg_reward),
            ]), global_step=agent.steps)

            agent.writer.add_summary(summary=tf.Summary(value=[
                tf.Summary.Value(tag="episode_reward", simple_value=total_r),
            ]), global_step=agent.steps)

            agent.writer.add_summary(summary=tf.Summary(value=[
                tf.Summary.Value(tag="steps per second", simple_value=
                agent.steps / (time.time() - start_time)),
            ]), global_step=agent.steps)

            agent.writer.add_summary(summary=tf.Summary(value=[
                tf.Summary.Value(tag="steps per second", simple_value=
                agent.steps / (time.time() - start_time)),
            ]), global_step=agent.steps)

            agent.writer.add_summary(summary=tf.Summary(value=[
                tf.Summary.Value(tag="ep_length", simple_value=ep_length),
            ]), global_step=agent.steps)

            agent.writer.add_summary(summary=tf.Summary(value=[
                tf.Summary.Value(tag="avg_ep_length", simple_value=
                (avg_ep_length * num_episodes + ep_length) / (num_episodes + 1) ),
            ]), global_step=agent.steps)

            total_r = 0
            num_episodes += 1
            ep_length = 0

        ep_length += 1

        hist_buffer = np.roll(hist_buffer, shift=-1, axis=2)
        hist_buffer[:, :, -1] = x #np.maximum(hist_buffer[:, :, -2], x)
        a = agent.act([hist_buffer / 255.0])
        x_prime, reward, done, _ = env.step(a)
        total_r += reward
        x_prime = preprocess(x_prime)
        agent.update(hist_buffer, a, reward, x_prime / 255.0, done)
        x = x_prime
        timestep += 1
        # imshow(np.concatenate([hist_buffer[:, :, 0], hist_buffer[:, :, 1],
        #                        hist_buffer[:, :, 2], hist_buffer[:, :, 2]]),
        #        cmap="gray")
        # pause(0.001)


env = gym.make("PongNoFrameskip-v4")
if "Videos" in os.listdir("."):
    shutil.rmtree(os.getcwd() + "\\Videos")
if "Models" in os.listdir("."):
    shutil.rmtree(os.getcwd() + "\\Models")
if "TensorBoardDir" in os.listdir("."):
    shutil.rmtree(os.getcwd() + "\\TensorBoardDir")
env = gym.wrappers.Monitor(env, "Videos", video_callable=lambda e_id: e_id % 15 == 0)
agent = C51Agent(num_actions=env.action_space.n)
train(env, agent, 5e7)

#show()