import numpy as np
import torch

from util import to_device
from config import MEM_NUM_CLASSES, SR_NUM_CLASSES


def generate_single_lable_memory_data(num_observations, sequence_length, time_delta=None):
    '''
    Generates num_observations sequences of length sequence_length where the entire sequence is 
    filled with 0s, except for one singular signal at a random position inside the sequence.
    The label of a sequence is the singular signal value. Because the NLLLoss used to train 
    the network expects labels in the range between 0 and num_classes - 1 instead of 1 and num_classes.
    '''
    size = ((num_observations, sequence_length, 1))
    data = np.zeros(size)    
    labels = np.zeros((num_observations, 1))
    for i, row in enumerate(data):
        signal = np.random.randint(1, MEM_NUM_CLASSES, 1)[0]
        last_possible_signal = sequence_length if not time_delta else sequence_length - time_delta
        column = np.random.randint(0, last_possible_signal, 1)[0]
        row[column] = signal
        labels[i] = signal - 1

    return to_device(torch.from_numpy(data).float()), to_device(torch.from_numpy(labels).long())


def generate_multi_lable_memory_data(num_observations, sequence_length, time_delta=None):
    '''
    Generates num_observations sequences of length sequence_length where the entire sequence is 
    filled with 0s, except for one singular signal at a random position inside the sequence.
    '''
    size = ((num_observations, sequence_length, 1))
    data = np.zeros(size)    
    labels = np.zeros((num_observations, sequence_length))
    for i, row in enumerate(data):
        signal = np.random.randint(1, MEM_NUM_CLASSES, 1)[0]
        last_possible_signal = sequence_length if not time_delta else sequence_length - time_delta
        column = np.random.randint(0, last_possible_signal, 1)[0]
        
        row[column] = signal
        labels[i, :-1] = 0
        labels[i, -1] = signal - 1

    return to_device(torch.from_numpy(data).float()), to_device(torch.from_numpy(labels).long())


def generate_store_and_recall_data(num_observations, sequence_length, recall_repetition=0, time_delta=None):
    '''
    Generates num_observations sequences of length sequence_length. The input
    consists of three different signals:
    1. Data signal: A stream of data points with a value between 1 and SR_NUM_CLASSES + 1
    2. Store signal: Either 0 if NO storage is required, or 1 if the current data point is supposed to
                     be stored by the network
    3. Recall signal: Either 0 if NO recall is required, or 1 if the last stored data point is supposed
                      to be recalled by the network.
    The label is a sequence of length num_observations that is 0 if the recall signal is 0 or has the value
    of the last stored data point if the corresponding recall signal is 1.
    '''
    size = ((num_observations, sequence_length, 1))
    data_stream = np.random.randint(1, SR_NUM_CLASSES - 1, size)
    store_signal = np.zeros(size)
    recall_signal = np.zeros(size)
    labels = np.zeros(size)
    data = []
    time_delta = time_delta if time_delta else 1

    for i, row in enumerate(data_stream):
        # select time step where store signal is sent
        store = np.random.choice(list(range(sequence_length - time_delta - recall_repetition)))
        store_signal[i][store] = 1

        # select time step where recall signal is sent
        recall = np.random.choice(list(range(store + time_delta, sequence_length - recall_repetition)))
        
        for j in range(0, recall_repetition + 1):
            recall_signal[i][recall + j] = 1
            labels[i][recall + j] = row[store]

        data.append(np.hstack((row, store_signal[i], recall_signal[i])))

    data = np.array(data)
    return to_device(torch.from_numpy(data).float()), to_device(torch.from_numpy(labels).long())