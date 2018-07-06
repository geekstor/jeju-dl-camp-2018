import random

import gym
import numpy as np
import tensorflow as tf
from tensorflow import layers

env_id = "CartPole-v0"
#env = Monitor(gym.make(env_id), video_callable=lambda x: x % 10 == 0,
#              directory="TestMountainCarVideo", force=True)

env = gym.make(env_id)

gamma = 0.99

from tensorflow.python.debug import TensorBoardDebugWrapperSession
sess = TensorBoardDebugWrapperSession(tf.Session(), "SB2-Valli:6064")

import matplotlib.pyplot as plt

class IQNNetwork:
    def __init__(self, scope, num_inputs, num_actions, N, trainable):
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        with tf.variable_scope(scope):
            # Size [Batch Size, *Raw X]
            self.state = tf.placeholder(name="state", shape=[None, self.num_inputs],
                                        dtype=tf.float32)

            self.outs = []
            self.outs.append(layers.dense(inputs=self.state, units=64,
                                          activation=tf.nn.relu, trainable=trainable))

            # Size [Batch Size, Feature Vec. Size]

            self.tau = tf.random_uniform(shape=[N,],
                                    minval=0, maxval=1,
                                    dtype=tf.float32)

            batch_tau = tf.tile(tf.expand_dims(self.tau, axis=0), [tf.shape(self.state)[0], 1])

            phi = tf.layers.dense(inputs=tf.cos(self.tau * tf.range(64, dtype=tf.float32) * 3.14159265), units=64,
                            activation=tf.nn.relu)

            mul = phi * self.outs[-1]

            self.y = tf.layers.dense(inputs=mul, units=num_actions, activation=None)

            self.out = tf.argmax(self.y, axis=-1)

class IQN:
    def __init__(self, num_inputs, num_actions):
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.train_net = IQNNetwork("train_base_net",
                                     num_inputs, num_actions, 8,  trainable=True)
        self.target_net = IQNNetwork("target_base_net",
                                      num_inputs, num_actions, 8, trainable=False)

        self.batch_dim_range = tf.constant(np.arange(32), dtype=tf.int32)

        # OPS. for computing target quantile dist.
        self.flat_indices_for_argmax_action_target_net = tf.stack([self.batch_dim_range,
                                                                   tf.cast(
                                                                       tf.reduce_max(self.target_net.y, axis=-1),
                                                                   dtype=tf.int32)], axis=1)

        self.sampled_return_of_greedy_actions_target_net = \
            tf.gather_nd(self.target_net.y,
                         self.flat_indices_for_argmax_action_target_net)

        self.r = tf.placeholder(shape=[None,], dtype=tf.float32)
        self.t = tf.placeholder(shape=[None,], dtype=tf.uint8)

        self.expected_quantiles = self.r[:, tf.newaxis] + gamma * \
                                  tf.cast(self.t[:, tf.newaxis], dtype=tf.float32) * \
                                  self.sampled_return_of_greedy_actions_target_net

        # OPS. for computing dist. from current train net
        self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.flat_indices_chosen_actions = tf.stack([self.batch_dim_range,
                                             self.action_placeholder], axis=1)
        self.dist_of_chosen_actions = tf.gather_nd(self.train_net.y,
                                                   self.flat_indices_chosen_actions)

        self.target = tf.placeholder(dtype=tf.float32, shape=[None,1])

        u = self.target - self.dist_of_chosen_actions
        k = 1
        huber_loss = 0.5 * tf.square(tf.clip_by_value(tf.abs(u), 0.0, k))
        huber_loss += k * (tf.abs(u) - tf.clip_by_value(tf.abs(u), 0.0, k))
        quantile_loss = tf.abs(self.train_net.tau - tf.cast((u < 0), tf.float32)) *\
                               huber_loss
        self.loss = tf.reduce_mean(quantile_loss)

    def act(self, x, eps):
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return sess.run(self.train_net.out,
                            feed_dict={self.train_net.state: x})[0]

    def get_expected_quantiles(self, x, r, t):
        return sess.run(self.expected_quantiles, feed_dict={self.target_net.state: x,
                                                            self.r: r,
                                                            self.t: t})

    def get_quantile_dist_of_chosen_actions(self, x, a):
        return sess.run(self.dist_of_chosen_actions, feed_dict={self.train_net.state: x,
                                                                self.action_placeholder: a})


iqn = IQN(env.observation_space.shape[0], env.action_space.n)


def get_tau(state, action, reward, next_state, terminal):
    expected_quants = iqn.get_expected_quantiles(next_state, reward, terminal)
    chosen_action_dist = iqn.get_quantile_dist_of_chosen_actions(state, action)
    tau_hat = np.linspace(0.0, 1.0 - 1. / num_quant, num_quant) + 0.5 / num_quant
    tau_hat = np.tile(tau_hat, (batch_sz, 1))
    sorted_quantiles = np.argsort(chosen_action_dist)
    tau = tau_hat[:, sorted_quantiles][np.arange(batch_sz), np.arange(batch_sz)]
    return expected_quants, tau


train_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_base_net')
target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_base_net')

assign_ops = []
for main_var, target_var in zip(sorted(train_variables, key=lambda x : x.name),
                                sorted(target_variables, key=lambda x: x.name)):
    if(main_var.name.replace("train_base_net", "") == target_var.name.replace("target_base_net", "")):
        assign_ops.append(tf.assign(target_var, main_var))

print("Copying Ops.:", len(assign_ops))

copy_operation = tf.group(*assign_ops)

from collections import deque
replay_buffer = deque(maxlen=50000)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
from baselines.common import tf_util
train_step = tf_util.minimize_and_clip(optimizer, iqn.loss, var_list=train_variables)

def train(x, a, r, x_p, t):
    xpc_qnt, tau = get_tau(x, a, r, x_p, t)
    return sess.run([iqn.loss, train_step],
                    feed_dict={iqn.train_net.state: x,
                               iqn.action_placeholder: a,
                               iqn.quantile_midpoints: tau,
                               iqn.target: xpc_qnt})

init = tf.global_variables_initializer()
sess.run(init)
sess.run(copy_operation)


def viz_dist(x):
    # Plot
    h = np.squeeze(sess.run(fetches=iqn.train_net.y,
                            feed_dict={iqn.train_net.state: x}))
    l, s = np.linspace(0, 1, num_quant, retstep=True)
    for i in range(h.shape[0]):
        plt.subplot(env.action_space.n, 1, i + 1)
        plt.bar(l - s / 2., height=h[i], width=s,
                color="brown", edgecolor="red", linewidth=0.5, align="edge")
    plt.pause(0.1)
    plt.gcf().clear()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

import math
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


num_frames = 10000

losses = []
all_rewards = []
episode_reward = 0

batch_sz = 32
state = env.reset()
for frame_idx in range(1, num_frames + 1):
    action = iqn.act(np.array([state]), epsilon_by_frame(frame_idx))
    #if len(all_rewards) % 10 == 0:
    #    viz_dist(np.array([state]))

    next_state, reward, done, _ = env.step(action)
    replay_buffer.appendleft([state, action, reward, next_state, done])

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        #print("Ep. Reward:", episode_reward)
        all_rewards.append(episode_reward)
        if len(all_rewards) % 10 == 0:
            print("Num Ep.:", len(all_rewards), "Num Steps:", sum(all_rewards),
                  "Mean Reward:", np.mean(all_rewards[-100:]), "Mean Loss:",
                  "Not yet trained!" if len(losses) <= 0 else np.mean(losses),
                  all_rewards[-10:])
            losses = []
        episode_reward = 0

    if len(replay_buffer) >= batch_sz:
        batch = random.sample(replay_buffer, batch_sz)

        x_batch = [i[0] for i in batch]
        a_batch = [i[1] for i in batch]
        r_batch = [i[2] for i in batch]
        x_p_batch = [i[3] for i in batch]
        t_batch = [not i[4] for i in batch]

        loss = train(x_batch, a_batch, r_batch, x_p_batch, t_batch)[0]
        losses.append(loss)

    #if frame_idx % 200 == 0:
    #    plot(frame_idx, all_rewards, losses)

    if frame_idx % 1000 == 0:
        sess.run(copy_operation)

