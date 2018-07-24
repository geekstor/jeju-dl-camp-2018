import random

import gym
import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor
from tensorflow import layers
from tensorflow.python.training.saver import Saver

env_id = "CartPole-v0"
env = Monitor(gym.make(env_id), directory="IQNDueling",
              video_callable=lambda ep_id: ep_id % 100 == 0, force=True)

# TODO: Try SoftMax exploration.

gamma = 0.99

sess = tf.Session()


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

            #self.calc_beta_tau = tf.placeholder_with_default(False, shape=[])

            self.is_acting = tf.placeholder_with_default(False, shape=[])

            self.tau = tf.random_uniform(shape=[tf.shape(self.state)[0], tf.cond(self.is_acting, lambda: 4*N, lambda: N)],
                                    minval=0, maxval=1,
                                    dtype=tf.float32)

            #with tf.variable_scope("sampling"):
            #    self.a = layers.dense(inputs=self.outs[-1], units=1, activation=tf.nn.sigmoid, trainable=trainable)

            #    self.b = layers.dense(inputs=self.outs[-1], units=1, activation=tf.nn.sigmoid, trainable=trainable)

            #self.transformed_tau = tf.clip_by_value(self.a, 0, 1) + \
            #                           self.tau * tf.clip_by_value(self.b, 0, 1 - self.a)

            phi = tf.layers.dense(inputs=tf.cos(tf.einsum('bn,j->bnj', self.tau,
                                                          tf.range(64, dtype=tf.float32)) * 3.14159265), units=64,
                            activation=tf.nn.relu)

            mul = tf.einsum('bnj,bj->bnj', phi, self.outs[-1])

            #self.a_dist = tf.transpose(
            #    tf.layers.dense(inputs=mul, units=num_actions, activation=None),
            #    perm=[0, 2, 1]
            #)

            #self.v = tf.layers.dense(inputs=self.outs[-1], units=1, activation=None)

            #self.q_dist = tf.identity(tf.expand_dims(self.v, axis=-1) + self.a_dist - \
            #              tf.expand_dims(tf.reduce_mean(self.a_dist, axis=1), axis=1), name="q_dist")

            self.q_dist = tf.transpose(
                tf.layers.dense(inputs=mul, units=num_actions, activation=None),
                perm=[0, 2, 1]
            )

            self.q = tf.reduce_mean(self.q_dist, axis=-1)

            self.q_softmax = tf.nn.softmax(self.q)

            self.out = tf.cast(tf.squeeze(tf.argmax(self.q, axis=-1)), dtype=tf.int32)

# net = IQNNetwork("test", 2, 2, 8, True)
# sess.run(tf.global_variables_initializer())
# print(sess.run(net.q_dist, feed_dict={net.state: np.array([[0, 1], [0.3, 0.7]])}))

class IQN:
    def __init__(self, num_inputs, num_actions):
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.train_net = IQNNetwork("train_base_net",
                                     num_inputs, num_actions, 8,  trainable=True)

        self.target_net = IQNNetwork("target_base_net",
                                     num_inputs, num_actions, 8, trainable=False)

        self.batch_dim_range = tf.range(tf.shape(self.train_net.state)[0], dtype=tf.int32)

        # OPS. for computing dist. from current train net
        self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.flat_indices_chosen_actions = tf.stack([self.batch_dim_range,
                                                     self.action_placeholder], axis=1)
        self.dist_of_chosen_actions = tf.gather_nd(self.train_net.q_dist,
                                                   self.flat_indices_chosen_actions)

        self.q_of_chosen_actions = tf.gather_nd(self.train_net.q,
                                                   self.flat_indices_chosen_actions)

        # self.return_placeholder = tf.placeholder(shape=[None, ], dtype=tf.float32)

        #self.sampling_loss = tf.losses.absolute_difference(self.q_of_chosen_actions,
        #                                                  self.return_placeholder)

        # OPS. for computing target quantile dist.
        self.flat_indices_for_argmax_action_target_net = tf.stack([self.batch_dim_range,
                                                                   self.target_net.out], axis=1)

        self.sampled_return_of_greedy_actions_target_net = \
            tf.gather_nd(self.target_net.q_dist,
                         self.flat_indices_for_argmax_action_target_net)

        self.r = tf.placeholder(shape=[None,], dtype=tf.float32)
        self.t = tf.placeholder(shape=[None,], dtype=tf.uint8)

        self.expected_quantiles = self.r[:, tf.newaxis] + gamma * \
                                  tf.cast(self.t[:, tf.newaxis], dtype=tf.float32) * \
                                  self.sampled_return_of_greedy_actions_target_net

        u = self.expected_quantiles[:, tf.newaxis, :] - self.dist_of_chosen_actions[:, :, tf.newaxis]
        k = 1
        huber_loss = 0.5 * tf.square(tf.clip_by_value(tf.abs(u), 0.0, k))
        huber_loss += k * (tf.abs(u) - tf.clip_by_value(tf.abs(u), 0.0, k))
        quantile_loss = tf.abs(tf.reshape(tf.tile(self.train_net.tau, [1, tf.shape(
            self.target_net.tau)[1]]), [-1, tf.shape(self.train_net.tau)[1],
                                     tf.shape(self.target_net.tau)[1]]) - tf.cast((u < 0), tf.float32)) *\
                               huber_loss
        self.loss = tf.reduce_mean(quantile_loss)

    def act(self, x, eps):
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return sess.run(self.train_net.out,
                            feed_dict={self.train_net.state: x, self.train_net.is_acting: False})
        #return np.random.choice(np.array(list(range(self.num_actions))), p=np.squeeze(sess.run(self.train_net.q_softmax,
        #                     feed_dict={self.train_net.state: x, self.train_net.calc_beta_tau: True, self.train_net.is_acting: True})))

    def get_expected_quantiles(self, x, r, t):
        return sess.run(self.expected_quantiles, feed_dict={self.target_net.state: x,
                                                            self.r: r,
                                                            self.t: t})

    def get_quantile_dist_of_chosen_actions(self, x, a):
        return sess.run(self.dist_of_chosen_actions, feed_dict={self.train_net.state: x,
                                                                self.action_placeholder: a})


iqn = IQN(env.observation_space.shape[0], env.action_space.n)


train_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_base_net')
target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_base_net')
sampling_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_base_net/sampling')
train_variables = [i for i in train_variables if i not in sampling_variables]
print(train_variables)

assign_ops = []
for main_var, target_var in zip(sorted(train_variables, key=lambda x : x.name),
                                sorted(target_variables, key=lambda x: x.name)):
    if(main_var.name.replace("train_base_net", "") == target_var.name.replace("target_base_net", "")):
        assign_ops.append(tf.assign(target_var, main_var))

print("Copying Ops.:", len(assign_ops))

copy_operation = tf.group(*assign_ops)

from collections import deque
replay_buffer = deque(maxlen=50000)
return_buffer = deque(maxlen=50000)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
from baselines.common import tf_util
train_step = tf_util.minimize_and_clip(optimizer, iqn.loss, var_list=train_variables)

#optimizer_sampling = tf.train.AdamOptimizer(learning_rate=1e-2)
#train_step_sampling = tf_util.minimize_and_clip(optimizer_sampling, iqn.sampling_loss,
#                                                var_list=sampling_variables)

def train(x, a, r=None, x_p=None, t=None, true_return=None):
    # if true_return is not None:
    #     return sess.run([iqn.sampling_loss, train_step_sampling], feed_dict={iqn.train_net.state: x,
    #                                                       iqn.action_placeholder: a,
    #                                                       iqn.return_placeholder: true_return,
    #                                                       iqn.train_net.is_acting: True,
    #                                                       iqn.train_net.calc_beta_tau: False})
    return sess.run([iqn.loss, train_step],
                    feed_dict={iqn.train_net.state: x,
                               iqn.action_placeholder: a,
                               iqn.r: r, iqn.t: t, iqn.target_net.state: x_p})

init = tf.global_variables_initializer()
sess.run(init)
sess.run(copy_operation)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

import math
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

saver = tf.train.Saver(var_list=train_variables + sampling_variables,
                       max_to_keep=100, keep_checkpoint_every_n_hours=1)

num_frames = 20000

losses = []
sampling_losses = []
all_rewards = []
episode_reward = 0

episode_reward_record = []

num_steps = 0

batch_sz = 32
state = env.reset()
for frame_idx in range(1, num_frames + 1):
    action = iqn.act(np.array([state]), epsilon_by_frame(frame_idx))
    #if len(all_rewards) % 10 == 0:
    #    viz_dist(np.array([state]))

    next_state, reward, done, _ = env.step(action)
    replay_buffer.append([state, action, reward, next_state, done])

    state = next_state
    episode_reward += reward
    episode_reward_record.append(reward)
    num_steps += 1

    if done:
        state = env.reset()
        #print("Ep. Reward:", episode_reward)
        all_rewards.append(episode_reward)
        if len(all_rewards) % 10 == 0:
            print("Num Ep.:", len(all_rewards), "Num Steps:", sum(all_rewards),
                  "Mean Reward:", np.mean(all_rewards[-100:]), "Mean Loss:",
                  "Not yet trained!" if len(losses) <= 0 else np.mean(losses),
                  "Mean Sampling Loss:", "Not yet trained!" if len(sampling_losses) <= 0
                  else np.mean(sampling_losses),
                  all_rewards[-10:])
            losses = []
            sampling_losses = []
        episode_reward = 0


        def alt(rewards, discount):
            """
            C[i] = R[i] + discount * C[i+1]
            signal.lfilter(b, a, x, axis=-1, zi=None)
            a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                                  - a[1]*y[n-1] - ... - a[N]*y[n-N]
            """
            r = rewards[::-1]
            a = [1, -discount]
            b = [1]
            from scipy import signal
            y = signal.lfilter(b, a, x=r)
            return y[::-1]
        true_discounted_returns = alt(episode_reward_record, 0.99)

        for i in range(1, len(episode_reward_record)):
            return_buffer.append([replay_buffer[-i][0], replay_buffer[-i][1],
                                  true_discounted_returns[-i]])

        episode_reward_record = []
        num_steps = 0

    if len(replay_buffer) >= batch_sz:
        batch = random.sample(replay_buffer, batch_sz)

        x_batch = [i[0] for i in batch]
        a_batch = [i[1] for i in batch]
        r_batch = [i[2] for i in batch]
        x_p_batch = [i[3] for i in batch]
        t_batch = [not i[4] for i in batch]

        loss, _ = train(x_batch, a_batch, r_batch, x_p_batch, t_batch)
        losses.append(loss)

    # if len(return_buffer) >= batch_sz:
    #     batch = random.sample(return_buffer, batch_sz)
    #
    #     x_batch = [i[0] for i in batch]
    #     a_batch = [i[1] for i in batch]
    #     return_batch = [i[2] for i in batch]
    #
    #     loss, _ = train(x_batch, a_batch, true_return=return_batch)
    #     sampling_losses.append(loss)

    #if frame_idx % 200 == 0:
    #    plot(frame_idx, all_rewards, losses)

    if frame_idx % 1000 == 0:
        sess.run(copy_operation)
        saver.save(sess=sess, global_step=frame_idx, save_path="IQNDueling/Model")

