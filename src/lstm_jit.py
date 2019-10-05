import torch
import torch.nn as nn
import torch.jit as jit

import math

from util import to_device
from eprop_func import EProp1


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


class BPTTCell(LSTMCell):
    """ 
    Custom LSTM Cell implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """    
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
    """
    Custom LSTM Cell implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
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


class EpropLSTM(jit.ScriptModule):
    """ 
    Custom LSTM implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(EpropLSTM, self).__init__()
        self.cell = EpropCell(input_size, hidden_size, bias)

    @jit.script_method
    def forward(self, input, initial_h, initial_c):
        # input (seq_len x batch_size x input_size)
        # initial_hidden (batch x hidden_size)
        # initial_state (batch x hidden_size)
        inputs = input.unbind(0)
        input_size = input.size(2)
        hidden_size = initial_h.size(1)
        batch_size = input.size(1)
        hx = initial_h
        cx = initial_c

        ev_w_ih_x = to_device(torch.zeros(batch_size, 3 * hidden_size, input_size))
        ev_w_hh_x = to_device(torch.zeros(batch_size, 3 * hidden_size, hidden_size))
        ev_b_x = to_device(torch.zeros(batch_size, 3 * hidden_size, 1))
        forgetgate = to_device(torch.zeros(batch_size, hidden_size, 1))

        outputs = []
        for i in range(len(inputs)):
            out, hx, cx, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate = self.cell(inputs[i], hx, cx, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate)
            outputs += [out]

        return torch.stack(outputs), hx, cx


class LSTM(jit.ScriptModule):
    """ 
    Custom LSTM implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.cell = BPTTCell(input_size, hidden_size, bias)

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
            self.lstm = LSTM(input_size, hidden_size, bias) 
            self.model_name = 'LSTM_BPTT'
            # self.lstm = nn.LSTM(input_size, hidden_size)
        elif cell_type == BaseNetwork.EPROP_1:
            self.lstm = EpropLSTM(input_size, hidden_size, bias)
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
        # lstm_out, _ = self.lstm(input)

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