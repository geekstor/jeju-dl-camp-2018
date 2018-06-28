import tensorflow as tf
GLOBAL_MANAGER = None

EPISODIC_LIFE = False
HISTORY_LEN = 4
SCALED_FLOAT = False
CLIP_REWARDS = True
MAX_TIMESTEPS = 2e6  # Number of Actions to Take.
MINIBATCH_SIZE = 128  # Per Update
EXPERIENCE_REPLAY_SIZE = 10000  # Num. Transitions
COPY_TARGET_FREQ = 250  # In Number of Updates.
UPDATE_FREQUENCY = 4  # Every u_f actions, an update is performed.
ACTION_REPEAT = 4  # Number of times to repeat each action.
LEARNING_RATE = 1e-4
EPSILON_ADAM = 1e-8
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_FINAL_STEP = 250000  # In number of actions (since learning).
NOOP_MAX = 30  # Execute up to NOOP_MAX noop actions at the beginning of an episode. # TODO: Use No-op Max.
MODEL_SAVE_FREQ = 1000 # In Number of updates.
VIDEOS_FOLDER = "Test\\Videos"
MODELS_FOLDER = "Test\\Models"
TENSORBOARD_FOLDER = "Test\\TensorBoardDir"
EPISODE_RECORD_FREQ = 15
REPLAY_START_SIZE = 10000
GYM_ENV_NAME = "PongNoFrameskip-v4"
ACTIONS_SPECIFICATION = [0, 2, 3]
STATE_DIMENSIONS = [84, 84, HISTORY_LEN]
STATE_DTYPE = tf.uint8
SHOW_IMAGES = False
CONVOLUTIONAL_LAYERS_SPEC = [
    {"filters": 32, "kernel_size": 8, "strides": 4},
    {"filters": 64, "kernel_size": 4, "strides": 2},
    {"filters": 64, "kernel_size": 3, "strides": 1},
]
DENSE_LAYERS_SPEC = [256]
NB_ATOMS = 51
V_MIN = -10
V_MAX = 10
GRAD_NORM_CLIP = 10.
MAX_MODELS_TO_KEEP = 10
MIN_MODELS_EVERY_N_HOURS = 2
NORMALIZED_STATE_NEEDED = True
DISCOUNT_FACTOR = 0.99
