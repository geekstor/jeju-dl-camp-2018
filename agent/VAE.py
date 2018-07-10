import tensorflow as tf
from matplotlib import pyplot as plt


def tf_func(q):
    sess = tf.Session()

    x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3])
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
                                   activation=tf.nn.sigmoid)

    latent_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                                      tf.log(tf.maximum(tf.square(sigma), 1e-10)) - 1, 1)

    latent_loss = tf.reduce_mean(tf.maximum(latent_loss, 0.5 * 64))

    reconstruction_loss = tf.losses.mean_squared_error(out, x)

    loss = latent_loss + reconstruction_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_step = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    import numpy as np

    while True:
        inp = []
        for i in range(100):
            inp.append(q.get())
        y, _ = sess.run([out, train_step], feed_dict={x: np.array(inp)})
        plt.subplot(2, 1, 1)
        plt.imshow(inp[0])
        plt.subplot(2, 1, 2)
        plt.imshow(y[0])

        plt.pause(0.1)


def run_env(q):
    import gym
    import random
    import skimage.transform

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
    _q = m.Queue()

    env_pool = []
    for i in range(16):
        env_pool.append(Process(target=run_env, args=(_q,)))

    p2 = Process(target=tf_func, args=(_q,))

    for i in range(16):
        env_pool[i].start()
    p2.start()

    for i in range(16):
        env_pool[i].join()
    p2.join()



