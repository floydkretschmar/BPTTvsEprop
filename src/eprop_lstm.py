import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F

import math

from util import to_device
from eprop_func2 import calculate_new_eligibility_vector, calculate_eligibility_trace


class Hook():
    def __init__(self, module, backward=False):
        if backward is False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


#class LSTM(jit.ScriptModule):
class LSTM(nn.Module):
    """ 
    Custom LSTM implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    # def __init__(self, input_size, hidden_size, bias=True):
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

    def calculate_custom_gradients(self, grad_loss_to_h):
        pass
    
    def _preprocess_forward(self, input, initial_h, initial_c):
        pass

    def _postprocess_cell(self, input, hx, cx, hy, cy, ingate, forgetgate_y, cellgate, outgate, time_step):
        pass

    #@jit.script_method
    def forward(self, input, initial_h, initial_c):
        # input (seq_len x batch_size x input_size)
        # initial_hidden (batch x hidden_size)
        # initial_state (batch x hidden_size)
        inputs = input.unbind(0)
        hx = initial_h
        cx = initial_c

        self._preprocess_forward(input, initial_h, initial_c)

        for i in range(len(inputs)):
            out, hx, cx = self.cell(inputs[i], hx, cx, i)

        return hx

    #@jit.script_method
    def cell(self, input, hx, cx, time_step):
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

        self._postprocess_cell(input, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, time_step)

        return hy, hy, cy


class EPropLSTM(LSTM):
    """ 
    Custom LSTM implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    # def __init__(self, input_size, hidden_size, bias=True):
    def __init__(self, input_size, hidden_size, bias=True):
        super(EPropLSTM, self).__init__(input_size, hidden_size, bias)

    def _preprocess_forward(self, input, initial_h, initial_c):
        batch_size = input.size(1)
        sequence_length = input.size(0)

        # Initialize eligibility traces and forgetgate to zero
        self.ev_w_ih = torch.zeros(sequence_length, batch_size, 4 * self.hidden_size, self.input_size)
        self.ev_w_hh = torch.zeros(sequence_length, batch_size, 4 * self.hidden_size, self.hidden_size)
        self.ev_b = torch.zeros(sequence_length, batch_size, 4 * self.hidden_size, 1)
        self.et_w_ih = torch.zeros(self.ev_w_ih.size())
        self.et_w_hh = torch.zeros(self.ev_w_hh.size())
        self.et_b = torch.zeros(self.ev_b.size())
        self.forgetgate = torch.zeros(batch_size, self.hidden_size, 1)

    def calculate_custom_gradients(self, grad_loss_to_h):
        sequence_length = self.et_w_ih.size(0)
        hidden_size = int(self.et_w_ih.size(2) / 4)
        grad_loss_to_h = grad_loss_to_h.unsqueeze(2)

        grad_weight_ih = torch.zeros(self.et_w_ih.size())
        grad_weight_hh = torch.zeros(self.et_w_hh.size())
        grad_bias = torch.zeros(self.et_b.size())
        
        # ingate, forgetgate and cellgate
        for i in range(0, 4 * hidden_size, hidden_size):
            grad_weight_ih[:, :, i:i + hidden_size, :] = self.et_w_ih[:, :, i:i + hidden_size, :] * grad_loss_to_h
            grad_weight_hh[:, :, i:i + hidden_size, :] = self.et_w_hh[:, :, i:i + hidden_size, :] * grad_loss_to_h
            grad_bias[:, :, i:i + hidden_size, :] = self.et_b[:, :, i:i + hidden_size, :] * grad_loss_to_h
            
        self.weight_ih.grad = grad_weight_ih.sum(dim=0).sum(dim=0)
        self.weight_hh.grad = grad_weight_hh.sum(dim=0).sum(dim=0)
        self.bias_ih.grad = grad_bias.sum(dim=0).sum(dim=0).squeeze()
        self.bias_hh.grad = grad_bias.sum(dim=0).sum(dim=0).squeeze()

    def _postprocess_cell(self, input, hx, cx, hy, cy, ingate, forgetgate_y, cellgate, outgate, time_step):
        if time_step == 0:
            ev_w_ih = torch.zeros(self.ev_w_ih[0].size())
            ev_w_hh = torch.zeros(self.ev_w_hh[0].size())
            ev_b = torch.zeros(self.ev_b[0].size())
        else:
            ev_w_ih = self.ev_w_ih[time_step - 1]
            ev_w_hh = self.ev_w_hh[time_step - 1]
            ev_b = self.ev_b[time_step - 1]

        self.ev_w_ih[time_step], self.ev_w_hh[time_step], self.ev_b[time_step] = calculate_new_eligibility_vector(
            ev_w_ih,
            ev_w_hh,
            ev_b,
            input,
            hx,
            ingate,
            self.forgetgate,
            forgetgate_y,
            cellgate,
            outgate,
            hy,
            cy)
        
        self.et_w_ih[time_step], self.et_w_hh[time_step], self.et_b[time_step] = calculate_eligibility_trace(
            self.ev_w_ih[time_step], 
            self.ev_w_hh[time_step], 
            self.ev_b[time_step],
            outgate
        )

        self.forgetgate = forgetgate_y.unsqueeze(2)

    
class MemoryLSTM(nn.Module):
    BPTT = 0
    EPROP_1 = 1
    EPROP_2 = 2
    EPROP_3 = 3

    def __init__(self, input_size, hidden_size, output_size, cell_type=BPTT, bias=True, batch_first=False, model_name='LSTM_BPTT'):
        super(MemoryLSTM, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.model_name = model_name
        self.batch_first = batch_first

        # LSTM layer
        if cell_type == MemoryLSTM.BPTT:
            self.lstm = LSTM(input_size, hidden_size, bias)
            # self.lstm = nn.LSTM(input_size, hidden_size)
        elif cell_type == MemoryLSTM.EPROP_1:
            self.lstm = EPropLSTM(input_size, hidden_size, bias)

        # LSTM to output mapping
        self.dense = nn.Linear(hidden_size, output_size, bias)
        self.lstm_output = None

    def calculate_custom_gradients(self):
        self.lstm.calculate_custom_gradients(self.lstm_output.grad)

    def forward(self, input):
        # prepare input and initial state
        if self.batch_first:
            input = input.permute(1, 0, 2)

        initial_h = to_device(torch.zeros(input.size(1), self.hidden_size))
        initial_c = to_device(torch.zeros(input.size(1), self.hidden_size))

        # lstm and dense pass for prediction
        self.lstm_output = self.lstm(input, initial_h, initial_c)
        # save final gradient
        self.lstm_output.retain_grad()

        # mapping to outputs
        dense_out = self.dense(self.lstm_output)
        predictions = F.log_softmax(dense_out, dim=1)
        return predictions

    def save(self, path, epoch):
        torch.save(self.state_dict(), '{}{}_{}.pth'.format(path, self.model_name, epoch, '.pth'))

    def load(self, path):
        self.load_state_dict(torch.load('{}/load/{}{}'.format(path, self.model_name, '.pth')))
        self.eval()
