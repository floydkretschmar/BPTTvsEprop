import argparse

from lstm_jit import MemoryLSTM
import torch
import torch.nn as nn
import torch.optim as optim
import time

from util import to_device
from memory_task import generate_data, test
import config


def get_batched_data(data, labels):
    permutation = torch.randperm(data.size()[0])
    for i in range(0, data.size()[0], config.BATCH_SIZE):
        indices = permutation[i:i + config.BATCH_SIZE]
        yield data[indices], labels[indices]


def main(args):
    # the default LSTM implementation with bptt
    model = to_device(MemoryLSTM(config.INPUT_SIZE, config.HIDEN_SIZE, config.NUM_CLASSES, cell_type=MemoryLSTM.BPTT, batch_first=True))

    if args.test:
        model.load(config.LOAD_PATH)

    if not args.test:
        # Use negative log-likelihood and ADAM
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        loss_function = nn.NLLLoss()

        train_X, train_Y = generate_data(config.TRAIN_SIZE, config.SEQ_LENGTH)
        training_results = []

        print('######################### TRAINING #########################')
        for epoch in range(1, config.NUM_EPOCHS + 1):
            start_time = time.time()
            total_loss = 0
            batch_num = 0

            for batch_x, batch_y in get_batched_data(train_X, train_Y):
                # reset gradient
                optimizer.zero_grad()
                prediction = model(batch_x)
                
                # Compute the loss, gradients, and update the parameters
                loss = loss_function(prediction, batch_y.squeeze())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_num += 1

            training_results.append('Epoch {} \t => Loss: {} [Batch-Time = {}s]'.format(epoch, total_loss / batch_num, round(time.time() - start_time, 2)))
            print(training_results[-1])

            if epoch % 50 == 0:
                model.save(config.SAVE_PATH, epoch)

        with open('{}/results.txt'.format(config.SAVE_PATH), 'w') as file:
            for result in training_results:
                file.write("{}\n".format(result))

    test(model, loss_function, config.TEST_SIZE, config.SEQ_LENGTH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, required=False)
    args = parser.parse_args()

    main(args)