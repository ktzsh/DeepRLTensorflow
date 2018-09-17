BATCH_SIZE = 64       # Batch Size
IMAGE_SIZE = (84, 84)  # Size of Image
FRAME_HISTORY = 4     # Number of frames history to keep
ACTION_REPEAT = 4     # Number of FRAME_SKIP
UPDATE_FREQ = 4       # number of new transitions to add to memory after...
# ...sampling a batch of transitions for training
EVAL_EPISODE = 50     # Number of epsiode after which evaluation is run
MAX_EPOCHS = 800      # Maximun Training Epochs

VIZ_SCALE = 0.01

GAMMA = 0.99

# Will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
MEMORY_SIZE = 1e6
INIT_MEMORY_SIZE = MEMORY_SIZE // 20     # Initial Memory Size
TARGET_NETWORK_UPDATE_FREQ = 10000 // UPDATE_FREQ
STEPS_PER_EPOCH = 100000 // UPDATE_FREQ  # Each epoch is 100k played frames
