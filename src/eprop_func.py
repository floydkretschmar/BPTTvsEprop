import torch
import torch.jit as jit
import time


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
        input = input.unsqueeze(2)
        hx = hx.unsqueeze(2)

        forgetgate_x = forgetgate_x.repeat(1, 3, 1)

        # the new eligibility vectors ...
        ev_w_ih_y = ev_w_ih_x.clone() * forgetgate_x
        ev_w_hh_y = ev_w_hh_x.clone() * forgetgate_x
        ev_b_y = ev_b_x.clone() * forgetgate_x

        # ... and eligibility traces
        et_w_ih_y = torch.Tensor(batch_size, 4 * hidden_size, input_size)
        et_w_hh_y = torch.Tensor(batch_size, 4 * hidden_size, hidden_size)
        et_b_y = torch.Tensor(batch_size, 4 * hidden_size, 1)

        ones = torch.ones(ingate.size())

        ds_dw_ih = torch.Tensor(ev_w_ih_y.size())
        ds_dw_hh = torch.Tensor(ev_w_hh_y.size())
        ds_db = torch.Tensor(ev_b_y.size())

        # ingate
        base = ingate * (ones - ingate) * cellgate
        ds_dw_hh[:, :hidden_size, :] = base * hx
        ds_dw_ih[:, :hidden_size, :] = base * input
        ds_db[:, :hidden_size, :] = base
        
        # forgetgate
        base = forgetgate_y * (ones - forgetgate_y) * cellgate
        ds_dw_hh[:, hidden_size:(2 * hidden_size), :] = base * hx
        ds_dw_ih[:, hidden_size:(2 * hidden_size), :] = base * input
        ds_db[:, hidden_size:(2 * hidden_size), :] = base

        # cellgate
        base = ingate * (ones - cellgate**2)
        ds_dw_hh[:, (2 * hidden_size):(3 * hidden_size), :] = base * hx
        ds_dw_ih[:, (2 * hidden_size):(3 * hidden_size), :] = base * input
        ds_db[:, (2 * hidden_size):(3 * hidden_size), :] = base

        ev_w_ih_y += ds_dw_ih
        ev_w_hh_y += ds_dw_hh
        ev_b_y += ds_db

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
        et_w_ih_y[:, (3 * hidden_size):(4 * hidden_size)] = base * hx
        et_w_hh_y[:, (3 * hidden_size):(4 * hidden_size)] = base * input
        et_b_y[:, (3 * hidden_size):(4 * hidden_size)] = base

        ctx.intermediate_results = et_w_ih_y, et_w_hh_y, et_b_y

        return hy, cy, ev_w_ih_y, ev_w_hh_y, ev_b_y, forgetgate_y

    @staticmethod
    # grad_ev_ih and grad_ev_hh should always be None
    def backward(ctx, grad_hy, grad_cy, grad_ev_w_ih, grad_ev_w_hh, grad_et_b, forgetgate_y):
        et_w_ih_y, et_w_hh_y, et_b_y = ctx.intermediate_results

        tmp_grad_hy = grad_hy.unsqueeze(2).repeat(1, 4, 1)

        #print(et_b_y.size(), tmp_grad_hy.size())

        grad_weight_ih = et_w_ih_y.clone() * tmp_grad_hy
        grad_weight_hh = et_w_hh_y.clone() * tmp_grad_hy
        grad_bias = et_b_y.clone() * tmp_grad_hy

        # create tensors for the weight gradient
        '''
        hidden_size = int(et_w_ih_y.size(1) / 4)
        grad_weight_ih = torch.Tensor(et_w_ih_y.size())
        grad_weight_hh = torch.Tensor(et_w_hh_y.size())
        grad_bias = torch.Tensor(et_b_y.size())

        grad_hy = grad_hy.unsqueeze(2)
        
        # ingate, forgetgate and cellgate
        for i in range(0, 4 * hidden_size, hidden_size):
            grad_weight_ih[:, i:i + hidden_size, :] = et_w_ih_y[:, i:i + hidden_size, :] * grad_hy
            grad_weight_hh[:, i:i + hidden_size, :] = et_w_hh_y[:, i:i + hidden_size, :] * grad_hy
            grad_bias[:, i:i + hidden_size, :] = et_b_y[:, i:i + hidden_size, :] * grad_hy

        #print(grad_weight_ih.shape, grad_weight_hh.shape, grad_bias.shape)
        grad_bias = grad_bias.squeeze()'''

        # grad_input, grad_ev_ih, grad_ev_hh, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh
        return None, None, None, None, None, None, None, grad_weight_ih, grad_weight_hh, grad_bias.squeeze(), grad_bias.squeeze()
