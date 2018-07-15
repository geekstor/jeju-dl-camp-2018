import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import gym
import random
import skimage.transform

def run_env(q):
    env = gym.make("PongNoFrameskip-v4")
    while True:
        done = False
        state = env.reset()
        while not done:
            a = random.randint(0, env.action_space.n - 1)
            next_state, r, done, _ = env.step(a)
            #q.put([skimage.transform.resize(state / 256.0, (64, 64), anti_aliasing=True), a, r,
            #               skimage.transform.resize(next_state / 256.0, (64, 64), anti_aliasing=True), done])
            q.put(skimage.transform.resize(state / 256.0, (64, 64), anti_aliasing=True))
            state = next_state


if __name__ == "__main__":
    from multiprocessing import Process
    from multiprocessing import Manager

    m = Manager()
    _q = m.Queue(10000)

    env_pool = []
    for i in range(16):
        env_pool.append(Process(target=run_env, args=(_q,)))

    for i in range(16):
        env_pool[i].start()

    sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1. / 16
    config.gpu_options.allow_growth = True

    x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name="x")
    h = tf.layers.conv2d(inputs=x, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu)
    h = tf.layers.conv2d(inputs=h, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
    h = tf.layers.conv2d(inputs=h, filters=128, kernel_size=4, strides=2, activation=tf.nn.relu)
    h = tf.layers.conv2d(inputs=h, filters=256, kernel_size=4, strides=2, activation=tf.nn.relu)

    h = tf.layers.flatten(h)

    mu = tf.layers.dense(inputs=h, units=64, activation=None)
    sigma = tf.layers.dense(inputs=h, units=64, activation=None)

    z = mu + tf.random_normal(shape=[tf.shape(mu)[0], 64]) * sigma

    h = tf.layers.dense(inputs=z, units=1024)

    h = tf.reshape(h, [-1, 1, 1, 1024])

    h = tf.layers.conv2d_transpose(inputs=h, filters=128, kernel_size=5, strides=2,
                                   activation=tf.nn.relu)
    h = tf.layers.conv2d_transpose(inputs=h, filters=64, kernel_size=5, strides=2,
                                   activation=tf.nn.relu)
    h = tf.layers.conv2d_transpose(inputs=h, filters=32, kernel_size=6, strides=2,
                                   activation=tf.nn.relu)
    out = tf.layers.conv2d_transpose(inputs=h, filters=3, kernel_size=6, strides=2,
                                     activation=tf.nn.sigmoid, name="out")

    latent_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                                      tf.log(tf.maximum(tf.square(sigma), 1e-10)) - 1, 1)

    latent_loss = tf.reduce_mean(tf.maximum(latent_loss, 0.5 * 64))

    reconstruction_loss = tf.losses.mean_squared_error(out, x)

    loss = latent_loss + reconstruction_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_step = optimizer.minimize(loss)

    from util.util import get_vars_with_scope
    saver = tf.train.Saver(var_list=get_vars_with_scope(""), max_to_keep=100, keep_checkpoint_every_n_hours=1)

    sess.run(tf.global_variables_initializer())

    from tqdm import tqdm
    for i in tqdm(range(0, 10000 * 1024, 256)):
        inp = []
        for _ in range(256):
            inp.append(_q.get_nowait())
        y, _ = sess.run([out, train_step], feed_dict={x: np.array(inp)})

        if i % 100 == 0:
            # saver.save(sess, "VAE/", global_step=i, write_meta_graph=True)

            plt.subplot(2, 1, 1)
            plt.imshow(inp[0])
            plt.subplot(2, 1, 2)
            plt.imshow(y[0])

            plt.pause(0.1)
            plt.savefig("Step%d.jpg" % i, dpi=300)
            plt.close()


