import numpy as np
import tensorflow as tf
from scipy.misc import imresize

sess = tf.Session()
new_saver = tf.train.import_meta_graph('~/Desktop/test/model.ckpt-660000.meta')
new_saver.restore(sess, "~/Desktop/test/model.ckpt-660000")

import gym
import random
print([n.name for n in tf.get_default_graph().as_graph_def().node])
from retro_contest.local import make
gym_env_required_params = ["GYM_ENV_NAME"]
gym_env_cfg = config_parser.parse_and_return_dictionary(
        "ENVIRONMENT", gym_env_required_params)
self.env = make(game=gym_env_cfg["GYM_ENV_NAME"], state=gym_env_cfg["GYM_ENV_LEVEL"], bk2dir=config_parser["TRAIN_FOLDER"])
from util.wrappers import wrap_env, SonicActionWrapper
if "WRAP_SONIC" in gym_env_cfg and gym_env_cfg["WRAP_SONIC"]:
    self.env = SonicActionWrapper(self.env)
self.env = wrap_env(self.env, gym_env_cfg)
state = env.reset()
buffer = []
for frame_idx in range(2000):
    action, dist = sess.run(fetches=["train_base_net/Cast:0", "train_base_net/q_dist:0"], feed_dict={"train_base_net/state:0": np.array([state])})
    image_state = imresize(env.render(mode="rgb_array"), [20, 30])

    buffer.append([image_state, dist])
    state, _, done, _ = env.step(action if random.random() > 0.1 else random.choice([0, 1]))

    if done:
        state = env.reset()
        print(frame_idx, " complete.")

from tensorboard.plugins import projector

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

images = [i[0] for i in buffer]
image_height, image_width, channels = images[0].shape
print(image_width, image_height, channels)
import math
images_per_row = int(math.ceil(math.sqrt(len(images))))
master_height = images_per_row * image_height
master_width = images_per_row * image_width
num_channels = channels
master = np.zeros([master_height, master_width, num_channels])
for idx, image in enumerate(images):
    left_idx = idx % images_per_row
    top_idx = int(math.floor(idx / images_per_row))
    left_start = left_idx * image_width
    left_end = left_start + image_width
    top_start = top_idx * image_height
    top_end = top_start + image_height
    master[top_start:top_end, left_start:left_end, :] = image

open("~/Desktop/images.png", "wb").write(
    tf.image.encode_png(master).eval(session=sess)
)

embedding_var = tf.Variable(np.reshape(np.array([i[1] for i in buffer]), [2000, -1]), name="actions")
sess.run(embedding_var.initializer)

print(sess.run(fetches=embedding_var).shape)

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.sprite.image_path = "images.png"
# Specify the width and height of a single thumbnail.
embedding.sprite.single_image_dim.extend([image_width, image_height])

summary_writer = tf.summary.FileWriter("./IQNDueling")

projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver([embedding_var])
saver.save(sess, "./IQNDueling/acts")
