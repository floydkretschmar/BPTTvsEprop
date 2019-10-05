import torch
import torch.nn as nn
import torch.jit as jit

import math

import lstm
from util import to_device
from eprop_func import EProp1


class BaseNetwork(nn.Module):
    BPTT = 0
    EPROP_1 = 1
    EPROP_2 = 2
    EPROP_3 = 3

    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 bias=True, 
                 batch_first=True, 
                 single_output=True,
                 cell_type=BPTT):
        super(BaseNetwork, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_first = batch_first
        self.single_output = single_output

        # LSTM layer
        if cell_type == BaseNetwork.BPTT:
            self.lstm = lstm.LSTM(input_size, hidden_size, lstm.BPTTCell(input_size, hidden_size, bias)) 
            self.model_name = 'LSTM_BPTT'
        elif cell_type == BaseNetwork.EPROP_1:
            self.lstm = lstm.EpropLSTM(input_size, hidden_size, lstm.EpropCell(input_size, hidden_size, bias))
            self.model_name = 'LSTM_EPROP'

        # LSTM to output mapping
        self.dense = nn.Linear(hidden_size, output_size, bias)

    def forward(self, input):
        # prepare input and initial state
        if self.batch_first:
            input = input.permute(1, 0, 2)

        initial_h = to_device(torch.zeros(input.size(1), self.hidden_size))
        initial_c = to_device(torch.zeros(input.size(1), self.hidden_size))

        # lstm and dense pass for prediction
        lstm_out, _, _ = self.lstm(input, initial_h, initial_c)

        # mapping to outputs
        if self.single_output:
            lstm_out = lstm_out[-1, :, :]       
        else:
            lstm_out = lstm_out.permute(1, 0, 2)
            
        dense_out = self.dense(lstm_out)    
        predictions = dense_out
        return predictions

    def save(self, path, epoch):
        torch.save(self.state_dict(), '{}{}_{}.pth'.format(path, self.model_name, epoch))

    def load(self, path):
        self.load_state_dict(torch.load('{}{}.pth'.format(path, self.model_name)))
        self.eval()


class MemoryNetwork(BaseNetwork):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 cell_type=BaseNetwork.BPTT):
        super(MemoryNetwork, self).__init__(input_size, hidden_size, output_size, cell_type=cell_type, single_output=True)


class StoreRecallNetwork(BaseNetwork):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 cell_type=BaseNetwork.BPTT):
        super(StoreRecallNetwork, self).__init__(input_size, hidden_size, output_size, cell_type=cell_type, single_output=False)