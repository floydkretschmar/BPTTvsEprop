import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F

import math

from util import to_device


class LSTMCell(jit.ScriptModule):
    """ 
    Custom LSTM Cell implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create for weight matrices, one for each gate + recurrent connections
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None

        # Initialize eligibility trace to zero
        self.eligibility_vector_ih = torch.zeros(hidden_size, input_size)
        self.eligibility_vector_hh = torch.zeros(hidden_size, hidden_size)

        self.initialize_parameters(self.weight_ih, self.bias_ih)
        self.initialize_parameters(self.weight_hh, self.bias_hh)
    
    def initialize_parameters(self, weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

    @jit.script_method
    def forward(self, input, hx, cx):
        # net activations...
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # ... and gate activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, hy, cy


class LSTM(jit.ScriptModule):
    """ 
    Custom LSTM implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.cell = LSTMCell(input_size, hidden_size, bias)

    @jit.script_method
    def forward(self, input, initial_h, initial_c):
        # input (seq_len x batch_size x input_size)
        # initial_hidden (batch x hidden_size)
        # initial_state (batch x hidden_size)
        inputs = input.unbind(0)
        hx = initial_h
        cx = initial_c

        outputs = []
        for i in range(len(inputs)):
            out, hx, cx = self.cell(inputs[i], hx, cx)
            outputs += [out]

        return torch.stack(outputs), hx, cx


class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=True, batch_first=False, model_name='LSTM_BPTT'):
        super(MemoryLSTM, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.model_name = model_name
        self.batch_first = batch_first

        # LSTM layer
        self.lstm = LSTM(input_size, hidden_size, bias)
        # self.lstm = nn.LSTM(input_size, hidden_size)

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
        # lstm_out, _ = self.lstm(input)

        # mapping to outputs
        dense_out = self.dense(lstm_out[-1, :, :])
        predictions = F.log_softmax(dense_out, dim=1)
        return predictions

    def save(self, path, epoch):
        torch.save(self.state_dict(), '{}{}_{}.pth'.format(path, self.model_name, epoch, '.pth'))

    def load(self, path):
        self.load_state_dict(torch.load('{}/load/{}{}'.format(path, self.model_name, '.pth')))
        self.eval()