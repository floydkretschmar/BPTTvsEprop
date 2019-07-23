import argparse

from lstm_jit import MemoryLSTM
from learning_task import MemoryTask

from util import to_device

# General Parameters #
SEQ_LENGTH = 700
TRAIN_SIZE = 10000
TEST_SIZE = 5000
NUM_EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 0.01
SAVE_PATH = './models/save/'
LOAD_PATH = './models/load/'

# Model parameters #
NUM_CLASSES = 9
INPUT_SIZE = 1
HIDEN_SIZE = 32


def main(args):
    task = MemoryTask(
        NUM_EPOCHS,
        BATCH_SIZE,
        SAVE_PATH,
        NUM_CLASSES)
    # the default LSTM implementation with bptt
    model = to_device(MemoryLSTM(INPUT_SIZE, HIDEN_SIZE, NUM_CLASSES, batch_first=True))

    if args.test:
        model.load(LOAD_PATH)

    if not args.test:
        task.train(model, TRAIN_SIZE, SEQ_LENGTH, LEARNING_RATE)

    task.test_all_deltas(model, TEST_SIZE, SEQ_LENGTH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, required=False)
    args = parser.parse_args()

    main(args)