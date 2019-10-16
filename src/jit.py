import torch
import torch.nn as nn
import torch.jit as jit

import math

from util import to_device
from lstm import LSTMCell
from eprop_func import forward_lstm, prepare_data, calculate_eligibility_trace, calculate_eligibility_vector

class Eprop1CellJit(LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(Eprop1CellJit, self).__init__(input_size, hidden_size, bias)

    @torch.jit.script_method
    def forward(self, input, hx, cx, ev_w_ih_x, ev_w_hh_x, ev_b_x, forgetgate):
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

    @torch.jit.autograd_script
    def eprop_func(self,
            ev_w_ih_x,
            ev_w_hh_x,
            ev_b_x,
            forgetgate_x,
            input_data, 
            hx, 
            cx,
            weight_ih, 
            weight_hh, 
            bias_ih=None, 
            bias_hh=None):
        ingate, forgetgate_y, cellgate, outgate, cy, hy = forward_lstm(weight_ih, weight_hh, bias_ih, bias_hh, input_data, hx, cx)

        # TODO: calculate new eligibility vector and trace
        # There exist distinct eligibility traces and vectors for the followiug parts of the LSTM cell:
        # - input to hidden connections, hidden to hidden connections, bias 
        # all for each: 
        # - ... inputgate, forgetgate and cellgate
        # => overall 3 * 3 = 9 eligibility traces
        input_data, ingate, forgetgate_y, forgetgate_x, cellgate, outgate, hx, hy, cx, ones, hidden_size, batch_size, input_size = prepare_data(
            input_data, ingate, forgetgate_y, forgetgate_x, cellgate, outgate, hx, hy, cx)

        ev_w_ih_y, ev_w_hh_y, ev_b_y = calculate_eligibility_vector(
            ones, 
            ev_w_ih_x, 
            ev_w_hh_x, 
            ev_b_x, 
            input_data, 
            ingate, 
            cellgate, 
            forgetgate_y, 
            forgetgate_x, 
            hx, 
            cx, 
            hidden_size)

        et_w_ih_y, et_w_hh_y, et_b_y = calculate_eligibility_trace(
            ones, 
            ev_w_ih_y, 
            ev_w_hh_y, 
            ev_b_y,
            input_data,
            outgate, 
            hx, 
            cy, 
            batch_size,
            input_size,
            hidden_size)

        def backward(grad_hy, grad_cy, grad_ev_w_ih, grad_ev_w_hh, grad_ev_b, grad_forgetgate_y):
            # Approximate dE/dh by substituting only with local error (\partial E)/(\partial h)
            tmp_grad_hy = grad_hy.unsqueeze(2).repeat(1, 4, 1)

            grad_weight_ih = et_w_ih_y * tmp_grad_hy
            grad_weight_hh = et_w_hh_y * tmp_grad_hy
            grad_bias = et_b_y * tmp_grad_hy

            # grad_ev_ih, grad_ev_hh, grad_ev_b, grad_forgetgate_x, grad_input, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh
            return None, None, None, None, None, None, None, grad_weight_ih, grad_weight_hh, grad_bias.squeeze(), grad_bias.squeeze()

        return hy, cy, ev_w_ih_y, ev_w_hh_y, ev_b_y, forgetgate_y, backward