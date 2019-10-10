import argparse

from models import BPTT_LSTM, EPROP1_LSTM, EPROP3_LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime

from util import to_device
from learning_tasks import generate_single_lable_memory_data, generate_store_and_recall_data
import config

import logging

STORE_RECALL = "S_R"
MEMORY = "MEM"

BPTT = "BPTT"
EPROP_1 = "EPROP1"
EPROP_3 = "EPROP3"


def chose_task(memory_task, training_algorithm):
    # Chose the task and corresponding model:
    if memory_task == MEMORY:
        generate_data = generate_single_lable_memory_data
        input_size = config.MEM_INPUT_SIZE
        hidden_size = config.MEM_HIDEN_SIZE 
        output_size = config.MEM_NUM_CLASSES
        single_output = True
        loss_function = nn.CrossEntropyLoss()
    elif memory_task == STORE_RECALL:
        generate_data = generate_store_and_recall_data
        input_size = config.SR_INPUT_SIZE
        hidden_size = config.SR_HIDEN_SIZE 
        output_size = config.SR_NUM_CLASSES + 1
        single_output = False
        loss_function = nn.CrossEntropyLoss()

    if training_algorithm == BPTT:
        model_constructor = BPTT_LSTM
        train_function = lambda optimizer, loss_func, batch_x, batch_y : train_bptt(optimizer, loss_func, batch_x, batch_y, memory_task)
    elif training_algorithm == EPROP_1:
        model_constructor = EPROP1_LSTM
        train_function = lambda optimizer, loss_func, batch_x, batch_y : train_bptt(optimizer, loss_func, batch_x, batch_y, memory_task)
    elif training_algorithm == EPROP_3:
        model_constructor = EPROP3_LSTM
        train_function = lambda optimizer, loss_func, batch_x, batch_y : train_eprop3(optimizer, loss_func, batch_x, batch_y, memory_task)
    
    model = to_device(model_constructor(
            input_size,
            hidden_size,
            output_size,
            single_output=single_output))
    return generate_data, model, loss_function, train_function


def format_pred_and_gt(pred, gt, memory_task):
    if memory_task == STORE_RECALL:
        pred = pred.view(-1, pred.size(2))
        gt = gt.clone().squeeze().view(-1)
    else:
        gt = gt.clone().squeeze()

    return pred, gt


def get_batched_data(data, labels):
    permutation = torch.randperm(data.size()[0])
    for i in range(0, data.size()[0], config.BATCH_SIZE):
        indices = permutation[i:i + config.BATCH_SIZE]
        yield data[indices], labels[indices]


def setup_logging(args):
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-8s %(message)s',
                        filename='{}/training_results.log'.format(config.SAVE_PATH))
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logging.info("----------------- Started Run -----------------")
    logging.info("Time: {}".format(datetime.now()))
    if args.training_algorithm == BPTT:
        ta = "BPTT"
    elif args.training_algorithm == EPROP_1:
        ta = "EPROP_1"

    if args.memory_task == MEMORY:
        task = "Memory"
    elif args.memory_task == STORE_RECALL:
        task = "Store and Recall"

    logging.info("Task: {}".format(task))   
    logging.info("Training algorithm: {}".format(ta))


def test(model, loss_function, generate_data, size_test_data, sequence_length, memory_task):
    logging.info('----------------- Started Testing -----------------')
    for delta in range(1, sequence_length):
        test_X, test_Y = generate_data(size_test_data, sequence_length, time_delta=delta)
        logging.info("Delta {}:".format(delta))
        with torch.no_grad():
            pred = model(test_X)
            prediction, gt = format_pred_and_gt(pred, test_Y, args.memory_task)
            loss = loss_function(prediction, gt)

            logging.info('Loss: {}'.format(loss))

def train_eprop3(optimizer, loss_function, batch_x, batch_y, memory_task):
    # reset gradient
    optimizer.zero_grad()

    prediction = model(batch_x)
    prediction, gt = format_pred_and_gt(prediction, batch_y, memory_task)
    
    # Compute the loss, gradients, and update the parameters
    loss = loss_function(prediction, gt)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_bptt(optimizer, loss_function, batch_x, batch_y, memory_task):
    # reset gradient
    optimizer.zero_grad()

    prediction = model(batch_x)
    prediction, gt = format_pred_and_gt(prediction, batch_y, memory_task)
    
    # Compute the loss, gradients, and update the parameters
    loss = loss_function(prediction, gt)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_truncated(optimizer, loss_function, batch_x, batch_y, training_func, truncation_delta):
    seq_len = batch_x.shape[1]

    total_loss = 0
    for i, start in enumerate(range(0, seq_len, truncation_delta)):
        # if truncated, select the current truncation part
        batch_x_part = batch_x[:,start:start+truncation_delta,:]
        batch_y_part = batch_y[:,start:start+truncation_delta,:]

        loss = training_func(batch_x, batch_y)
        total_loss += loss

    # reset model if necessary
    model.reset()
    return total_loss / i


def train(model, generate_data, loss_function, train_func, truncation_delta=0):
    # Use negative log-likelihood and ADAM for training
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # data generation is dependend on the training task
    train_X, train_Y = generate_data(config.TRAIN_SIZE, config.SEQ_LENGTH)
    training_results = []

    logging.info('----------------- Started Training -----------------')
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        total_loss = 0
        batch_num = 0

        for batch_x, batch_y in get_batched_data(train_X, train_Y):
            if truncation_delta > 0:
                total_loss += train_truncated(
                    optimizer, 
                    loss_function, 
                    batch_x, 
                    batch_y, 
                    lambda trunc_batch_x, trunc_batch_y : train_func(optimizer, loss_function, trunc_batch_x, trunc_batch_y), 
                    truncation_delta)
            else:
                total_loss += train_func(optimizer, loss_function, batch_x, batch_y)

            batch_num += 1

        result = 'Epoch {} \t => Loss: {} [Batch-Time = {}s]'.format(epoch, total_loss / batch_num, round(time.time() - start_time, 2))
        logging.info(result)
        
        if epoch % 25 == 0:
            logging.info("Saved model")
            model.save(config.SAVE_PATH, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('-m', '--memory_task', default=MEMORY, type=str)
    parser.add_argument('-a', '--training_algorithm', default=BPTT, type=str)
    args = parser.parse_args()

    setup_logging(args)

    generate_data, model, loss_function, train_function = chose_task(args.memory_task, args.training_algorithm)

    if args.test:
        model.load(config.LOAD_PATH)

    if not args.test:
        train(model, generate_data, loss_function, train_function, truncation_delta=config.TRUNCATION_DELTA)

    test(model, loss_function, generate_data, config.TEST_SIZE, config.SEQ_LENGTH, args.memory_task)
    
    logging.info("----------------- Finished Run -----------------")