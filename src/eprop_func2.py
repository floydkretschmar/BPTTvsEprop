import torch


def calculate_new_eligibility_vector(
        ev_w_ih_x,
        ev_w_hh_x,
        ev_b_x,
        input, 
        hx, 
        ingate,
        forgetgate_x, 
        forgetgate_y, 
        cellgate, 
        outgate,
        hy,
        cy):
    # There exist distinct eligibility traces and vectors for the followiug parts of the LSTM cell:
    # - input to hidden connections, hidden to hidden connections, bias 
    # all for each: 
    # - ... inputgate, forgetgate and cellgate
    # => overall 3 * 3 = 9 eligibility traces
    hidden_size = hy.size(1)

    # the new eligibility vectors ...
    ev_w_ih_y = torch.Tensor(ev_w_ih_x.size())
    ev_w_hh_y = torch.Tensor(ev_w_hh_x.size())
    ev_b_y = torch.Tensor(ev_b_x.size())

    ev_w_ih_i, ev_w_ih_f, ev_w_ih_c, _ = ev_w_ih_x.chunk(4, 1)
    ev_w_hh_i, ev_w_hh_f, ev_w_hh_c, _ = ev_w_hh_x.chunk(4, 1)
    ev_b_i, ev_b_f, ev_b_c, _ = ev_b_x.chunk(4, 1)

    ones = torch.ones(ingate.size())

    # ingate
    base = ingate * (ones - ingate) * cellgate
    ds_dw_hh_i = base * hx
    ds_dw_ih_i = base * input
    ds_dbias_i = base
    
    ev_w_ih_y[:, :hidden_size, :] = forgetgate_x * ev_w_ih_i + ds_dw_hh_i.unsqueeze(2)
    ev_w_hh_y[:, :hidden_size, :] = forgetgate_x * ev_w_hh_i + ds_dw_ih_i.unsqueeze(2)
    ev_b_y[:, :hidden_size, :] = forgetgate_x * ev_b_i + ds_dbias_i.unsqueeze(2)
    
    # forgetgate
    base = forgetgate_y * (ones - forgetgate_y) * cellgate
    ds_dw_hh_f = base * hx
    ds_dw_ih_f = base * input
    ds_dbias_f = base
    
    ev_w_ih_y[:, hidden_size:(2 * hidden_size), :] = forgetgate_x * ev_w_ih_f + ds_dw_hh_f.unsqueeze(2)
    ev_w_hh_y[:, hidden_size:(2 * hidden_size), :] = forgetgate_x * ev_w_hh_f + ds_dw_ih_f.unsqueeze(2)
    ev_b_y[:, hidden_size:(2 * hidden_size), :] = forgetgate_x * ev_b_f + ds_dbias_f.unsqueeze(2)
    
    # cellgate
    base = ingate * (ones - cellgate**2)
    ds_dw_hh_c = base * hx
    ds_dw_ih_c = base * input
    ds_dbias_c = base
    
    ev_w_ih_y[:, (2 * hidden_size):(3 * hidden_size), :] = forgetgate_x * ev_w_ih_c + ds_dw_hh_c.unsqueeze(2)
    ev_w_hh_y[:, (2 * hidden_size):(3 * hidden_size), :] = forgetgate_x * ev_w_hh_c + ds_dw_ih_c.unsqueeze(2)
    ev_b_y[:, (2 * hidden_size):(3 * hidden_size), :] = forgetgate_x * ev_b_c + ds_dbias_c.unsqueeze(2)

    # The gradient of the output gate is only dependent on the observable state
    # => just use normal gradient calculation of dE/dh * dh/dweight 
    # => calculate second part of that equation now for input to hidden, hidden to hidden 
    #    and bias connections and multiply in the backward pass
    base = outgate * (ones - outgate) * cy
    ev_w_ih_y[:, (3 * hidden_size):(4 * hidden_size)] = (base * hx).unsqueeze(2)
    ev_w_hh_y[:, (3 * hidden_size):(4 * hidden_size)] = (base * input).unsqueeze(2)
    ev_b_y[:, (3 * hidden_size):(4 * hidden_size)] = base.unsqueeze(2)

    return ev_w_ih_y, ev_w_hh_y, ev_b_y


def calculate_eligibility_trace(
        ev_w_ih,
        ev_w_hh,
        ev_b,
        outgate):
    # eligibility traces
    et_w_ih = ev_w_ih.clone()
    et_w_hh = ev_w_hh.clone()
    et_b = ev_b.clone()

    hidden_size = outgate.size(1)

    # calculate eligibility traces by multiplying the eligibility vectors with the outgate
    for i in range(0, 3 * hidden_size, hidden_size):
        et_w_ih[:, i:(i + hidden_size), :] *= outgate.unsqueeze(2)
        et_w_hh[:, i:(i + hidden_size), :] *= outgate.unsqueeze(2)
        et_b[:, i:(i + hidden_size), :] *= outgate.unsqueeze(2)

    return et_w_ih, et_w_hh, et_b
