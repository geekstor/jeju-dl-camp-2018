import random

import gym
import numpy as np
import tensorflow as tf
from tensorflow import layers

env_id = "CartPole-v0"
from gym.wrappers import Monitor
env = Monitor(gym.make(env_id), video_callable=lambda x: x % 10 == 0,
              directory="TestMountainCarVideo", force=True)

gamma = 0.99

sess = tf.Session()

num_quant = 51

import matplotlib.pyplot as plt

class BaseNetwork:
    def __init__(self, scope, num_inputs, num_actions, num_quantiles, trainable):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        with tf.variable_scope(scope):
            self.state = tf.placeholder(name="state", shape=(None, num_inputs), dtype=tf.float32)

            # Util.
            self.batch_dim_range = tf.constant(np.arange(32), dtype=tf.int32)
            self.num_quantiles_range = tf.constant(np.arange(num_quantiles), dtype=tf.int32)

            self.outs = []
            self.outs.append(layers.dense(inputs=self.state, units=64,
                                          activation=tf.nn.relu, trainable=trainable))
            self.outs.append(layers.dense(inputs=self.outs[-1],
                                          units=num_actions*num_quantiles,
                                          activation=None, trainable=trainable))
            self.out = tf.reshape(self.outs[-1], [-1, self.num_actions, self.num_quantiles])

            self.q_vals = tf.reduce_mean(self.out, axis=-1)

            self.greedy_action = tf.cast(tf.argmax(self.q_vals, axis=-1), dtype=tf.int32)

    def get_quantile_locs(self, x):
        return sess.run(self.out, feed_dict={self.state: x})

    def get_q_values(self, x):
        return sess.run(self.q_vals, feed_dict={self.state: x})

    def get_greedy_action(self, x):
        return sess.run(self.greedy_action, feed_dict={self.state: x})

class QRDQN:
    def __init__(self, num_inputs, num_actions, num_quantiles):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.train_net = BaseNetwork("train_base_net",
                                     num_inputs, num_actions, num_quantiles, trainable=True)
        self.target_net = BaseNetwork("target_base_net",
                                      num_inputs, num_actions, num_quantiles, trainable=False)


        # OPS. for computing target quantile dist.
        self.flat_indices_for_argmax_action_target_net = tf.stack([self.target_net.batch_dim_range,
                                                                   self.target_net.greedy_action], axis=1)
        self.dist_of_greedy_actions_target_net = \
            tf.reshape(tf.gather_nd(self.target_net.out,
                                    self.flat_indices_for_argmax_action_target_net),
                       [-1, self.num_quantiles])

        self.r = tf.placeholder(shape=[None,], dtype=tf.float32)
        self.t = tf.placeholder(shape=[None,], dtype=tf.uint8)

        self.expected_quantiles = self.r[:, tf.newaxis] + gamma * \
                                  tf.cast(self.t[:, tf.newaxis], dtype=tf.float32) * \
                                  self.dist_of_greedy_actions_target_net

        # OPS. for computing dist. from current train net
        self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.flat_indices_chosen_actions = tf.stack([self.train_net.batch_dim_range,
                                             self.action_placeholder], axis=1)
        self.dist_of_chosen_actions = tf.reshape(tf.gather_nd(self.train_net.out,
                                                   self.flat_indices_chosen_actions),
                                                 [-1, self.num_quantiles])

        # Compute loss.
        self.quantile_midpoints = tf.placeholder(dtype=tf.float32, shape=[None,
                                                                          self.num_quantiles])
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, self.num_quantiles])

        u = self.target - self.dist_of_chosen_actions
        k = 1
        huber_loss = 0.5 * tf.square(tf.clip_by_value(tf.abs(u), 0.0, k))
        huber_loss += k * (tf.abs(u) - tf.clip_by_value(tf.abs(u), 0.0, k))
        quantile_loss = tf.abs(self.quantile_midpoints - tf.cast((u < 0), tf.float32)) *\
                               huber_loss
        self.loss = tf.reduce_sum(quantile_loss) / num_quant

    def act(self, x, eps):
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return sess.run(self.train_net.greedy_action,
                            feed_dict={self.train_net.state: x})[0]

    def get_expected_quantiles(self, x, r, t):
        return sess.run(self.expected_quantiles, feed_dict={self.target_net.state: x,
                                                            self.r: r,
                                                            self.t: t})

    def get_quantile_dist_of_chosen_actions(self, x, a):
        return sess.run(self.dist_of_chosen_actions, feed_dict={self.train_net.state: x,
                                                                self.action_placeholder: a})

qrdqn = QRDQN(env.observation_space.shape[0], env.action_space.n, num_quant)


def get_tau(state, action, reward, next_state, terminal):
    expected_quants = qrdqn.get_expected_quantiles(next_state, reward, terminal)
    chosen_action_dist = qrdqn.get_quantile_dist_of_chosen_actions(state, action)
    tau_hat = np.linspace(0.0, 1.0 - 1. / num_quant, num_quant) + 0.5 / num_quant
    tau = np.tile(tau_hat, (batch_sz, 1))
    #sorted_quantiles = np.argsort(chosen_action_dist)
    #tau = tau_hat[:, sorted_quantiles][np.arange(batch_sz), np.arange(batch_sz)]
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
train_step = tf_util.minimize_and_clip(optimizer, qrdqn.loss, var_list=train_variables)

def train(x, a, r, x_p, t):
    xpc_qnt, tau = get_tau(x, a, r, x_p, t)
    return sess.run([qrdqn.loss, train_step],
                            feed_dict={qrdqn.train_net.state: x,
                                       qrdqn.action_placeholder: a,
                                       qrdqn.quantile_midpoints: tau,
                                       qrdqn.target: xpc_qnt})

init = tf.global_variables_initializer()
sess.run(init)
sess.run(copy_operation)


def viz_dist(x):
    # Plot
    h = np.squeeze(sess.run(fetches=qrdqn.train_net.out,
                                 feed_dict={qrdqn.train_net.state: x}))
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
    action = qrdqn.act(np.array([state]), epsilon_by_frame(frame_idx))
    if len(all_rewards) % 10 == 0:
        viz_dist(np.array([state]))

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

