import torch
import torch.jit as jit
import time

from util import to_device


class EProp1(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            ev_w_ih_x,
            ev_w_hh_x,
            ev_b_x,
            forgetgate_x,
            input, 
            hx, 
            cx,
            weight_ih, 
            weight_hh, 
            bias_ih=None, 
            bias_hh=None):
        gates = (torch.mm(input, weight_ih.t()) + bias_ih + torch.mm(hx, weight_hh.t()) + bias_hh)
        ingate, forgetgate_y, cellgate, outgate = gates.chunk(4, 1)

        # ... and gate activations
        ingate = torch.sigmoid(ingate)
        forgetgate_y = torch.sigmoid(forgetgate_y)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate_y * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        # TODO: calculate new eligibility vector and trace
        # There exist distinct eligibility traces and vectors for the followiug parts of the LSTM cell:
        # - input to hidden connections, hidden to hidden connections, bias 
        # all for each: 
        # - ... inputgate, forgetgate and cellgate
        # => overall 3 * 3 = 9 eligibility traces
        hidden_size = hy.size(1)
        batch_size = input.size(0)
        input_size = input.size(1)

        ingate = ingate.unsqueeze(2)
        forgetgate_y = forgetgate_y.unsqueeze(2)
        cellgate = cellgate.unsqueeze(2)
        outgate = outgate.unsqueeze(2)
        input = input.unsqueeze(1)
        hx = hx.unsqueeze(2)

        forgetgate_x = forgetgate_x.repeat(1, 3, 1)

        # the new eligibility vectors ...
        ev_w_ih_y = ev_w_ih_x.clone() * forgetgate_x
        ev_w_hh_y = ev_w_hh_x.clone() * forgetgate_x
        ev_b_y = ev_b_x.clone() * forgetgate_x

        # ... and eligibility traces
        et_w_ih_y = to_device(torch.Tensor(batch_size, 4 * hidden_size, input_size))
        et_w_hh_y = to_device(torch.Tensor(batch_size, 4 * hidden_size, hidden_size))
        et_b_y = to_device(torch.Tensor(batch_size, 4 * hidden_size, 1))

        ones = to_device(torch.ones(ingate.size()))

        # ingate
        base = ingate * (ones - ingate) * cellgate
        ev_w_hh_y[:, :hidden_size, :] += base * hx
        ev_w_ih_y[:, :hidden_size, :] += base * input
        ev_b_y[:, :hidden_size, :] += base
        
        # forgetgate
        base = forgetgate_y * (ones - forgetgate_y) * cellgate
        ev_w_hh_y[:, hidden_size:(2 * hidden_size), :] += base * hx
        ev_w_ih_y[:, hidden_size:(2 * hidden_size), :] += base * input
        ev_b_y[:, hidden_size:(2 * hidden_size), :] += base

        # cellgate
        base = ingate * (ones - cellgate**2)
        ev_w_hh_y[:, (2 * hidden_size):(3 * hidden_size), :] += base * hx
        ev_w_ih_y[:, (2 * hidden_size):(3 * hidden_size), :] += base * input
        ev_b_y[:, (2 * hidden_size):(3 * hidden_size), :] += base

        # calculate eligibility traces by multiplying the eligibility vectors with the outgate
        tmp_outgate = outgate.repeat(1, 3, 1)
        et_w_ih_y[:, :3 * hidden_size, :] = ev_w_ih_y * tmp_outgate
        et_w_hh_y[:, :3 * hidden_size, :] = ev_w_hh_y * tmp_outgate
        et_b_y[:, :3 * hidden_size, :] = ev_b_y * tmp_outgate
        
        # The gradient of the output gate is only dependent on the observable state
        # => just use normal gradient calculation of dE/dh * dh/dweight 
        # => calculate second part of that equation now for input to hidden, hidden to hidden 
        #    and bias connections and multiply in the backward pass
        base = outgate * (ones - outgate) * cy.unsqueeze(2)
        et_w_hh_y[:, (3 * hidden_size):(4 * hidden_size)] = base * hx
        et_w_ih_y[:, (3 * hidden_size):(4 * hidden_size)] = base * input
        et_b_y[:, (3 * hidden_size):(4 * hidden_size)] = base

        ctx.intermediate_results = et_w_ih_y, et_w_hh_y, et_b_y

        return hy, cy, ev_w_ih_y, ev_w_hh_y, ev_b_y, forgetgate_y

    @staticmethod
    # grad_ev_ih and grad_ev_hh should always be None
    def backward(ctx, grad_hy, grad_cy, grad_ev_w_ih, grad_ev_w_hh, grad_et_b, forgetgate_y):
        #print(grad_hy)

        et_w_ih_y, et_w_hh_y, et_b_y = ctx.intermediate_results

        tmp_grad_hy = grad_hy.unsqueeze(2).repeat(1, 4, 1)

        grad_weight_ih = et_w_ih_y * tmp_grad_hy
        grad_weight_hh = et_w_hh_y * tmp_grad_hy
        grad_bias = et_b_y * tmp_grad_hy

        # grad_input, grad_ev_ih, grad_ev_hh, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh
        return None, None, None, None, None, None, None, grad_weight_ih, grad_weight_hh, grad_bias.squeeze(), grad_bias.squeeze()
