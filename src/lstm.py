import torch
import torch.nn as nn
import torch.jit as jit

import math

from util import to_device
from autograd import ForgetGate

""" 
Custom LSTM implementation using jit adopted from:
https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
"""


class LSTM(nn.Module):
#class LSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
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

        self.initialize_parameters(self.weight_ih, self.bias_ih)
        self.initialize_parameters(self.weight_hh, self.bias_hh)

    def initialize_parameters(self, weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
    
    
class BPTT_LSTM(LSTM):
    def __init__(self, input_size, hidden_size, bias=True):
        super(BPTT_LSTM, self).__init__(input_size, hidden_size, bias)

    def forward(self, input, initial_h, initial_c):
        # input (seq_len x batch_size x input_size)
        # initial_hidden (batch x hidden_size)
        # initial_state (batch x hidden_size)
        inputs = input.unbind(0)
        hx = initial_h
        cx = initial_c
        outputs = []
        for i in range(len(inputs)):
            hx, cx = self.cell(inputs[i], hx, cx)
            outputs += [hx]

        return torch.stack(outputs), cx

    #@jit.script_method
    def cell(self, input, hx, cx):
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

        return hy, cy


class EPropLSTM(LSTM):
    def __init__(self, input_size, hidden_size, eprop_func, bias=True):
        super(EPropLSTM, self).__init__(input_size, hidden_size, bias)
        self.eprop_func = eprop_func

    def forward(
            self, 
            input, 
            initial_h, 
            initial_c,
            eligibility_vectors=[]):
        # input (seq_len x batch_size x input_size)
        # initial_hidden (batch x hidden_size)
        # initial_state (batch x hidden_size)
        inputs = input.unbind(0)
        input_size = input.size(2)
        hidden_size = initial_h.size(1)
        batch_size = input.size(1)
        hx = initial_h
        cx = initial_c

        if len(eligibility_vectors) == 0:
            ev_w_ih_x = to_device(torch.zeros(batch_size, 3 * hidden_size, input_size, requires_grad=False))
            ev_w_hh_x = to_device(torch.zeros(batch_size, 3 * hidden_size, hidden_size, requires_grad=False))
            ev_b_x = to_device(torch.zeros(batch_size, 3 * hidden_size, 1, requires_grad=False))
        else:
            ev_w_ih_x = eligibility_vectors[0]
            ev_w_hh_x = eligibility_vectors[1]
            ev_b_x = eligibility_vectors[2]

        forgetgate = to_device(torch.zeros(batch_size, hidden_size, 1))

        outputs = []
        for i in range(len(inputs)):
            hx, cx, ev_w_ih_x, ev_w_hh_x, ev_b_x, new_forgetgate = self.cell(inputs[i], hx, cx, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate)
            _, forgetgate = ForgetGate.apply(forgetgate, new_forgetgate)
            outputs += [hx]

        return torch.stack(outputs), cx, [ev_w_ih_x, ev_w_hh_x, ev_b_x]

    def cell(self, input, hx, cx, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate):
        hy, cy, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate = self.eprop_func(
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

        return hy, cy, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate