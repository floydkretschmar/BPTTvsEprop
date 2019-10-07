import torch
import torch.nn as nn
import torch.jit as jit

import math

import lstm
from util import to_device
from eprop_func import EProp1


class BaseNetwork(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 lstm,
                 bias=True, 
                 batch_first=True, 
                 single_output=True):
        super(BaseNetwork, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_first = batch_first
        self.single_output = single_output

        self.lstm = lstm

        # LSTM to output mapping
        self.dense = nn.Linear(hidden_size, output_size, bias)

    def forward(self, input):
        # prepare input and initial state
        if self.batch_first:
            input = input.permute(1, 0, 2)

        initial_h = to_device(torch.zeros(input.size(1), self.hidden_size))
        initial_c = to_device(torch.zeros(input.size(1), self.hidden_size))

        # lstm and dense pass for prediction
        lstm_out, _, _ = self.lstm(input, initial_h.detach(), initial_c.detach())

        # mapping to outputs
        if self.single_output:
            lstm_out = lstm_out[-1, :, :]       
        else:
            lstm_out = lstm_out.permute(1, 0, 2)
            
        dense_out = self.dense(lstm_out)    
        predictions = dense_out
        return predictions

    def get_name(self):
        return "BaseNetwork"

    def save(self, path, epoch):
        torch.save(self.state_dict(), '{}{}_{}.pth'.format(path, self.get_name(), epoch))

    def load(self, path):
        self.load_state_dict(torch.load('{}{}.pth'.format(path, self.get_name())))
        self.eval()

class BPTT_LSTM(BaseNetwork):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 bias=True, 
                 batch_first=True, 
                 single_output=True):
        super(BPTT_LSTM, self).__init__(
            input_size, 
            hidden_size, 
            output_size, 
            lstm.LSTM(input_size, hidden_size, lstm.BPTTCell(input_size, hidden_size, bias)), 
            bias=bias, 
            batch_first=batch_first, 
            single_output=single_output)
            
    def get_name(self):        
        return "LSTM_BPTT"


class EPROP1_LSTM(BaseNetwork):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 bias=True, 
                 batch_first=True, 
                 single_output=True):
        super(EPROP1_LSTM, self).__init__(
            input_size, 
            hidden_size, 
            output_size, 
            lstm.EpropLSTM(input_size, hidden_size, lstm.EpropCell(input_size, hidden_size, eprop_func=EProp1.apply, bias=bias)), 
            bias=bias, 
            batch_first=batch_first, 
            single_output=single_output)

    def get_name(self):        
        return "LSTM_EPROP1"


'''class EPROP1_LSTM(BaseNetwork):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 bias=True, 
                 batch_first=True, 
                 single_output=True):
        super(EPROP1_LSTM, self).__init__(
            input_size, 
            hidden_size, 
            output_size, 
            lstm.EpropLSTM(input_size, hidden_size, lstm.EpropCell(input_size, hidden_size, eprop_func=EProp1.apply, bias=bias)), 
            bias=bias, 
            batch_first=batch_first, 
            single_output=single_output)

    def get_name(self):        
        return "LSTM_EPROP1"'''