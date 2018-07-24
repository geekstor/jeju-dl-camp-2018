import random

import gym
import numpy as np
import tensorflow as tf
from tensorflow import layers

env_id = "CartPole-v0"
env = gym.make(env_id)

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

            self.calc_beta_tau = tf.placeholder_with_default(False, shape=[])

            self.is_acting = tf.placeholder_with_default(False, shape=[])

            self.use_tau_placeholder = tf.placeholder_with_default(False, shape=[])

            self.tau_placeholder = tf.placeholder_with_default(np.zeros(shape=[1, 51], dtype=np.float32), shape=[None, 51])

            self._tau = tf.random_uniform(shape=[tf.shape(self.state)[0], tf.cond(self.is_acting, lambda: 4*N, lambda: N)],
                                    minval=0, maxval=1,
                                    dtype=tf.float32)

            self.tau = tf.cond(self.use_tau_placeholder, lambda: self.tau_placeholder, lambda: self._tau)

            with tf.variable_scope("sampling"):
                self.a = layers.dense(inputs=self.outs[-1], units=1,
                                 activation=tf.nn.sigmoid,
                                 trainable=trainable)
                self.b = tf.clip_by_value(
                    layers.dense(inputs=self.outs[-1], units=1,
                                 activation=None,
                                 trainable=trainable), 0, 1 - self.a
                )

            self.transformed_tau = tf.stop_gradient(self.tau) * self.a

            phi = tf.layers.dense(inputs=tf.cos(tf.einsum('bn,j->bnj', tf.cond(self.calc_beta_tau, lambda: self.transformed_tau,
                                                                               lambda: self.tau),
                                                          tf.range(64, dtype=tf.float32)) * 3.1415926535), units=64,
                            activation=tf.nn.relu)

            mul = tf.einsum('bnj,bj->bnj', phi, self.outs[-1])

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

        self.return_placeholder = tf.placeholder(shape=[None, ], dtype=tf.float32)

        self.sampling_loss = tf.losses.absolute_difference(self.q_of_chosen_actions,
                                                          self.return_placeholder)

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
                            feed_dict={self.train_net.state: x, self.train_net.calc_beta_tau: True,
                                       self.train_net.is_acting: True})
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

optimizer_sampling = tf.train.AdamOptimizer(learning_rate=1e-2)
train_step_sampling = tf_util.minimize_and_clip(optimizer_sampling, iqn.sampling_loss,
                                                var_list=sampling_variables)

def train(x, a, r=None, x_p=None, t=None, true_return=None):
    if true_return is not None:
        return sess.run([iqn.sampling_loss, train_step_sampling], feed_dict={iqn.train_net.state: x,
                                                          iqn.action_placeholder: a,
                                                          iqn.return_placeholder: true_return,
                                                          iqn.train_net.is_acting: True})
    return sess.run([iqn.loss, train_step],
                    feed_dict={iqn.train_net.state: x,
                               iqn.action_placeholder: a,
                               iqn.r: r, iqn.t: t, iqn.target_net.state: x_p,
                               iqn.target_net.calc_beta_tau: True})


def viz_dist(x, rgb_x):
    tau = np.linspace(0, 1, 51)

    a, h = sess.run(fetches=[iqn.train_net.a[0],
                          iqn.train_net.q_dist[0]], feed_dict={iqn.train_net.state: x,
                                                      iqn.train_net.use_tau_placeholder: True,
                                                      iqn.train_net.tau_placeholder: [tau]})

    #print(h.shape)

    from matplotlib import pyplot as plt
    plt.subplot2grid((3, 3), (0, 0), colspan=1, rowspan=2)
    # plt.subplot(len(self.train_network.actions), 2, [1, 3])
    from scipy.misc import imresize
    plt.imshow(imresize(rgb_x, [rgb_x.shape[0] * 10, rgb_x.shape[1]]),
               aspect="auto", interpolation="nearest")

    for i in range(h.shape[0]):
        plt.subplot2grid((2, 2), (i, 1), colspan=1, rowspan=1)
        # plt.subplot(len(self.train_network.actions), 2, 2 * (i + 1)))
        plt.stem(tau, h[i], markerfmt=" ")
        plt.plot(a, 0, 'ko')

    plt.pause(0.1)
    plt.gcf().clear()

    data = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

    return data


init = tf.global_variables_initializer()
sess.run(init)
sess.run(copy_operation)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

import math
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


num_frames = 20000

losses = []
sampling_losses = []
all_rewards = []
episode_reward = 0

episode_reward_record = []

num_steps = 0

batch_sz = 32
state = env.reset()
render_buffer = []
return_buffer = []
ep_id = 1
for frame_idx in range(1, num_frames + 1):
    action = iqn.act(np.array([state]), epsilon_by_frame(frame_idx))
    if ep_id % 50 == 0:
        out = viz_dist([state], env.render(mode="rgb_array"))
        render_buffer.append(out)

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

        if len(render_buffer) > 0:
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip(render_buffer, fps=5)
            clip.write_gif("IQNViz3/ep" + str(ep_id) + '.gif', fps=5)
            render_buffer = []

        episode_reward_record = []
        num_steps = 0
        ep_id += 1

    if len(replay_buffer) >= batch_sz:
        batch = random.sample(replay_buffer, batch_sz)

        x_batch = [i[0] for i in batch]
        a_batch = [i[1] for i in batch]
        r_batch = [i[2] for i in batch]
        x_p_batch = [i[3] for i in batch]
        t_batch = [not i[4] for i in batch]

        loss, _ = train(x_batch, a_batch, r_batch, x_p_batch, t_batch)
        losses.append(loss)

    if len(return_buffer) >= batch_sz:
        batch = random.sample(return_buffer, batch_sz)

        x_batch = [i[0] for i in batch]
        a_batch = [i[1] for i in batch]
        return_batch = [i[2] for i in batch]

        loss, _ = train(x_batch, a_batch, true_return=return_batch)
        sampling_losses.append(loss)

    #if frame_idx % 200 == 0:
    #    plot(frame_idx, all_rewards, losses)

    if frame_idx % 1000 == 0:
        sess.run(copy_operation)

