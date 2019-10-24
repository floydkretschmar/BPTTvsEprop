import argparse

from models import BPTT_LSTM, EPROP1_LSTM, EPROP3_LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime

from util import to_device#, prepare_parameter_lists, copy_master_parameters_to_model, copy_model_gradients_to_master
from learning_tasks import generate_single_lable_memory_data, generate_store_and_recall_data
import config

import logging
import amp

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
        train_function = lambda model, optimizer, loss_func, batch_x, batch_y : train_bptt(model, optimizer, loss_func, batch_x, batch_y, memory_task)
    elif training_algorithm == EPROP_1:
        model_constructor = EPROP1_LSTM
        train_function = lambda model, optimizer, loss_func, batch_x, batch_y : train_bptt(model, optimizer, loss_func, batch_x, batch_y, memory_task)
    elif training_algorithm == EPROP_3:
        model_constructor = lambda in_size, h_size, o_size, single_output : EPROP3_LSTM(
            in_size, 
            h_size, 
            o_size, 
            single_output=single_output)
        train_function = lambda model, optimizer, loss_func, batch_x, batch_y : train_eprop3(
            model,
            optimizer, 
            loss_func, 
            batch_x, 
            batch_y, 
            memory_task, 
            config.TRUNCATION_DELTA)
    
    model = to_device(model_constructor(
            input_size,
            hidden_size,
            output_size,
            single_output=single_output))
    return generate_data, model, loss_function, train_function


def format_pred_and_gt(pred, gt, memory_task):
    if memory_task == STORE_RECALL:
        pred = pred.view(-1, pred.size(2))
        gt = gt.squeeze().flatten()
    else:
        gt = gt.squeeze()

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
    elif args.training_algorithm == EPROP_3:
        ta = "EPROP_3"

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

def train_eprop3(model, optimizer, loss_function, batch_x, batch_y, memory_task, truncation_delta):
    seq_len = batch_x.shape[1]
    loss_function_synth = nn.MSELoss()

    # reset model
    model.reset()

    # implementation of algo on page 33 of eprop paper
    for i, start in enumerate(range(truncation_delta, seq_len, truncation_delta)):
        # reset gradient
        model.zero_grad()

        # select [t_{m-1}+1, ..., t_{m}]
        first_batch_x = batch_x[:,start-truncation_delta:start,:].clone()
        first_batch_y = batch_y[:,start-truncation_delta:start,:].clone()

        # simulate network over [t_{m-1}+1, ..., t_{m}] and backprop using the synthetic gradient
        prediction, _, first_synth_grad = model(first_batch_x)

        pred, gt = format_pred_and_gt(prediction, first_batch_y, memory_task)  
        loss = loss_function(pred, gt)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # select [t_{m}+1, ..., t_{m+1}] (the next truncated time interval)
        second_batch_x = batch_x[:,start:start+truncation_delta,:].clone()
        second_batch_y = batch_y[:,start:start+truncation_delta,:].clone()

        # simulate and backprop using second synthetic gradient
        prediction, second_initial_state, second_synth_grad = model(second_batch_x)

        # retain grad of the initial hidden state of the second interval ...
        second_initial_state.retain_grad()

        pred, gt = format_pred_and_gt(prediction, second_batch_y, memory_task)    
        loss = loss_function(pred, gt) 
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # ... and store it ...
        real_grad_x = second_initial_state.grad.detach()  

        # ... to optimize the synth grad network using MSE
        loss = loss_function_synth(first_synth_grad, real_grad_x)
        scaled_loss = config.LOSS_SCALE_FACTOR * loss.float()   
        scaled_loss.backward()

        real_grad_x_shape = real_grad_x.shape

        # train the final synthetic gradient to be close to 0
        if start+truncation_delta == seq_len:
            zeros = to_device(torch.zeros(real_grad_x_shape, requires_grad=False))
            loss = loss_function_synth(second_synth_grad, zeros)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        optimizer.step()

    with torch.no_grad():
        prediction, _, _ = model(batch_x)
        pred, gt = format_pred_and_gt(prediction, batch_y, memory_task)
        loss = loss_function(pred, gt)

    return loss.item()

def train_bptt(model, optimizer, loss_function, batch_x, batch_y, memory_task):
    prediction = model(batch_x)
    prediction, gt = format_pred_and_gt(prediction, batch_y, memory_task)
    # Compute the loss, gradients, and update the parameters
    loss = loss_function(prediction, gt)
    
    # reset gradient
    model.zero_grad()
    
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    optimizer.step()
    return loss.item()


def train(model, generate_data, loss_function, train_func):
    # Use negative log-likelihood and ADAM for training
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # data generation is dependend on the training task
    train_X, train_Y = generate_data(config.TRAIN_SIZE, config.SEQ_LENGTH)
    training_results = []

    logging.info('----------------- Started Training -----------------')
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        total_loss = 0
        batch_num = 0

        for batch_x, batch_y in get_batched_data(train_X, train_Y):
            total_loss += train_func(model, optimizer, loss_function, batch_x, batch_y)
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
        train(model, generate_data, loss_function, train_function)

    test(model, loss_function, generate_data, config.TEST_SIZE, config.SEQ_LENGTH, args.memory_task)
    
    logging.info("----------------- Finished Run -----------------")