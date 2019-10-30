import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from models import BPTT_LSTM, EPROP1_LSTM, EPROP3_LSTM
from util import to_device, load_checkpoint
from learning_tasks import generate_single_lable_memory_data, generate_store_and_recall_data
from training import train, train_bptt, train_eprop3, get_batched_data, format_pred_and_gt
import config

BPTT = "BPTT"
EPROP_1 = "EPROP1"
EPROP_3 = "EPROP3"

def chose_task(memory_task, training_algorithm):
    # Chose the task and corresponding model:
    if memory_task == config.MEMORY:
        generate_data = generate_single_lable_memory_data
        input_size = config.MEM_INPUT_SIZE
        hidden_size = config.MEM_HIDEN_SIZE 
        output_size = config.MEM_NUM_CLASSES
        single_output = True
        loss_function = nn.CrossEntropyLoss()
    elif memory_task == config.STORE_RECALL:
        generate_data = generate_store_and_recall_data
        input_size = config.SR_INPUT_SIZE
        hidden_size = config.SR_HIDEN_SIZE 
        output_size = config.SR_NUM_CLASSES + 1
        single_output = False
        loss_function = nn.CrossEntropyLoss()

    if training_algorithm == BPTT:
        model_constructor = BPTT_LSTM
        train_function = lambda model, optimizer, loss_func, batch_x, batch_y : train_bptt(model, optimizer, loss_func, batch_x, batch_y)
    elif training_algorithm == EPROP_1:
        model_constructor = EPROP1_LSTM
        train_function = lambda model, optimizer, loss_func, batch_x, batch_y : train_bptt(model, optimizer, loss_func, batch_x, batch_y)
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
            config.TRUNCATION_DELTA)
    
    model = to_device(model_constructor(
            input_size,
            hidden_size,
            output_size,
            single_output=single_output))
    return generate_data, model, loss_function, train_function


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

    if args.memory_task == config.MEMORY:
        task = "Memory"
    elif args.memory_task == config.STORE_RECALL:
        task = "Store and Recall"

    logging.info("Task: {}".format(task))   
    logging.info("Training algorithm: {}".format(ta))


def test(model, loss_function, generate_data, size_test_data, sequence_length, single_output):
    logging.info('----------------- Started Testing -----------------')
    for delta in range(1, sequence_length):
        test_X, test_Y = generate_data(size_test_data, sequence_length, time_delta=delta)
        logging.info("Delta {}:".format(delta))
        with torch.no_grad():
            pred = model(test_X)
            prediction, gt = format_pred_and_gt(pred, test_Y, single_output)
            loss = loss_function(prediction, gt)

            logging.info('Loss: {}'.format(loss))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('-m', '--memory_task', default=config.MEMORY, type=str)
    parser.add_argument('-a', '--training_algorithm', default=BPTT, type=str)
    args = parser.parse_args()

    setup_logging(args)

    generate_data, model, loss_function, train_function = chose_task(args.memory_task, args.training_algorithm)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if args.test:
        load_checkpoint(model, optimizer)

    if not args.test:
        train(model, optimizer, generate_data, loss_function, train_function)

    test(model, loss_function, generate_data, config.TEST_SIZE, config.SEQ_LENGTH, model.single_output)
    
    logging.info("----------------- Finished Run -----------------")