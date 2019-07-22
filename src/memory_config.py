# Data parameters #
NUM_CLASSES = 9
SEQ_LENGTH = 200
TRAIN_SIZE = 10000
TEST_SIZE = 1000

# during inference: How many time steps between memory signal and inference?
SEQ_MIN_DELTA = 30

# Model parameters #
INPUT_SIZE = 1
HIDEN_SIZE = 32
NUM_EPOCHS = 300
BATCH_SIZE = 128

SAVE_PATH = './models/save/'
LOAD_PATH = './models/load/'
