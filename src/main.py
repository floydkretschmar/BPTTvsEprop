import argparse

from lstm_jit import MemoryLSTM
import torch
import torch.nn as nn
import torch.optim as optim
import time

from util import to_device
from memory_task import generate_data as mem_generate_data
from store_recall_task import generate_data as sr_generate_data
import config

STORE_RECALL = 'S_R'
MEMORY = 'MEM'


def get_batched_data(data, labels):
    permutation = torch.randperm(data.size()[0])
    for i in range(0, data.size()[0], config.BATCH_SIZE):
        indices = permutation[i:i + config.BATCH_SIZE]
        yield data[indices], labels[indices]


def format_pred_and_gt(pred, gt, mem_task):
    if mem_task == STORE_RECALL:
        pred = pred.view(-1, pred.size(2))
        gt = gt.squeeze().view(-1)
    elif mem_task == MEMORY:
        gt = gt.squeeze()

    return pred, gt


def test(model, loss_function, generate_data, size_test_data, sequence_length):
    print('######################### TESTING #########################')
    for delta in range(1, sequence_length):
        test_X, test_Y = generate_data(size_test_data, sequence_length, delta)
        print("Delta {}:".format(delta))
        with torch.no_grad():
            prediction = model(test_X)
            loss = loss_function(prediction, test_Y.squeeze())
            print('Loss: {}'.format(loss))


def chose_task(memory_task):
    # Chose the task and corresponding model:
    if memory_task == MEMORY:
        generate_data = mem_generate_data
        model = to_device(MemoryLSTM(
            config.MEM_INPUT_SIZE, 
            config.MEM_HIDEN_SIZE, 
            config.MEM_NUM_CLASSES))
        loss_function = nn.NLLLoss()
    elif memory_task == STORE_RECALL:
        generate_data = sr_generate_data
        model = to_device(MemoryLSTM(
            config.SR_INPUT_SIZE, 
            config.SR_HIDEN_SIZE, 
            config.SR_NUM_CLASSES + 1,
            single_output=False))
        loss_function = nn.NLLLoss()

    return generate_data, model, loss_function


def main(args):
    generate_data, model, loss_function = chose_task(args.mem_task)

    if args.test:
        model.load(config.LOAD_PATH)

    if not args.test:
        # Use negative log-likelihood and ADAM for training
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # data generation is dependend on the training task
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

                prediction, gt = format_pred_and_gt(prediction, batch_y, args.mem_task)
                
                # Compute the loss, gradients, and update the parameters
                loss = loss_function(prediction, gt)
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

    test(model, loss_function, generate_data, config.TEST_SIZE, config.SEQ_LENGTH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, required=False)
    parser.add_argument('--mem_task', default=MEMORY, type=str)
    args = parser.parse_args()

    main(args)