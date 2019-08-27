import numpy as np
import torch

from util import to_device
import config


def generate_data(num_observations, sequence_length, time_delta=None):
    '''
    Generates the sequences of form (num_observations, sequence_length) where the entire sequence is 
    filled with 0s, except for one singular signal at a random position inside the sequence.
    The label of a sequence is the true class of the signal - 1, because the NLLLoss used to train 
    the network expects labels in the range between 0 and num_classes - 1 instead of 1 and num_classes.
    '''
    size = ((num_observations, sequence_length, 1))
    data = np.zeros(size)    
    labels = np.zeros((num_observations, 1))
    for i, row in enumerate(data):
        signal = np.random.randint(1, config.NUM_CLASSES, 1)[0]
        last_possible_signal = sequence_length if not time_delta else sequence_length - time_delta
        column = np.random.randint(0, last_possible_signal, 1)[0]
        row[column] = signal
        labels[i] = signal - 1

    return to_device(torch.from_numpy(data).float()), to_device(torch.from_numpy(labels).long())


def test(model, loss_function, size_test_data, sequence_length):
    print('######################### TESTING #########################')
    for delta in range(1, sequence_length):
        test_X, test_Y = generate_data(size_test_data, sequence_length, delta)
        print("Delta {}:".format(delta))
        with torch.no_grad():
            prediction = model(test_X)
            loss = loss_function(prediction, test_Y.squeeze())
            print('Loss: {}'.format(loss))