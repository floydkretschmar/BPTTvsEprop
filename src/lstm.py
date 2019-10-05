import torch
import torch.nn as nn
import torch.jit as jit

import math

from util import to_device
from eprop_func import EProp1

""" 
Custom LSTM implementation using jit adopted from:
https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
"""

class LSTMCell(jit.ScriptModule):
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


class BPTTCell(LSTMCell):
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


class EpropCell(LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(EpropCell, self).__init__(input_size, hidden_size, bias)

    def forward(self, input, hx, cx, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate):
        hy, cy, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate = EProp1.apply(
            ev_w_ih_x,
            ev_w_hh_x,
            ev_b_x,
            forgetgate,
            input, 
            hx, 
            cx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh)

        return hy, hy, cy, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate


class LSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, cell):
        super(LSTM, self).__init__()
        self.cell = cell

    def forward(self, input, initial_h, initial_c):
        # input (seq_len x batch_size x input_size)
        # initial_hidden (batch x hidden_size)
        # initial_state (batch x hidden_size)
        return self.forward_core(input, initial_h, initial_c)

    def forward_core(self, inputs, hx, cx):
        inputs = inputs.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, hx, cx = self.cell(inputs[i], hx, cx)
            outputs += [out]

        return torch.stack(outputs), hx, cx


class EpropLSTM(LSTM):
    def forward_core(self, inputs, hx, cx):
        # input (seq_len x batch_size x input_size)
        # initial_hidden (batch x hidden_size)
        # initial_state (batch x hidden_size)
        input_size = inputs.size(2)
        hidden_size = hx.size(1)
        batch_size = inputs.size(1)
        inputs = inputs.unbind(0)

        ev_w_ih_x = to_device(torch.zeros(batch_size, 3 * hidden_size, input_size))
        ev_w_hh_x = to_device(torch.zeros(batch_size, 3 * hidden_size, hidden_size))
        ev_b_x = to_device(torch.zeros(batch_size, 3 * hidden_size, 1))
        forgetgate = to_device(torch.zeros(batch_size, hidden_size, 1))

        outputs = []
        for i in range(len(inputs)):
            out, hx, cx, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate = self.cell(inputs[i], hx, cx, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate)
            outputs += [out]

        return torch.stack(outputs), hx, cx