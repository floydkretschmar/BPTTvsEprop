import torch
import torch.nn as nn
import logging
import time

import config
from util import save_checkpoint, to_device

def format_pred_and_gt(pred, gt, single_output):
    if not single_output:
        pred = pred.view(-1, pred.size(2))
        gt = gt.squeeze().flatten()
    else:
        gt = gt.squeeze()

    return pred, gt


def get_batched_data(data, labels):
    permutation = torch.randperm(data.size()[0])
    for i in range(0, data.size()[0], config.BATCH_SIZE):
        indices = permutation[i:i + config.BATCH_SIZE]
        yield data[indices], labels[indices]


def train_eprop3(model, optimizer, loss_function, batch_x, batch_y, truncation_delta):
    seq_len = batch_x.shape[1]
    loss_function_synth = nn.MSELoss()

    # reset model
    model.reset()

    # implementation of algo on page 33 of eprop paper
    for i, start in enumerate(range(truncation_delta, seq_len, truncation_delta)):
        # reset gradient
        model.zero_grad()

        # select [t_{m-1}+1, ..., t_{m}]
        first_batch_x = batch_x[:,start-truncation_delta:start,:].clone()
        first_batch_y = batch_y[:,start-truncation_delta:start,:].clone()

        # simulate network over [t_{m-1}+1, ..., t_{m}] and backprop using the synthetic gradient
        prediction, _, first_synth_grad = model(first_batch_x)

        pred, gt = format_pred_and_gt(prediction, first_batch_y, model.single_output)  
        loss = loss_function(pred, gt)
        loss.backward()

        # select [t_{m}+1, ..., t_{m+1}] (the next truncated time interval)
        second_batch_x = batch_x[:,start:start+truncation_delta,:].clone()
        second_batch_y = batch_y[:,start:start+truncation_delta,:].clone()

        # simulate and backprop using second synthetic gradient
        prediction, second_initial_state, second_synth_grad = model(second_batch_x)

        # retain grad of the initial hidden state of the second interval ...
        second_initial_state.retain_grad()

        pred, gt = format_pred_and_gt(prediction, second_batch_y, model.single_output)    
        loss = loss_function(pred, gt) 
        loss.backward()

        # ... and store it ...
        real_grad_x = second_initial_state.grad.detach()  

        # ... to optimize the synth grad network using MSE
        loss = loss_function_synth(first_synth_grad, real_grad_x)
        loss.backward()

        real_grad_x_shape = real_grad_x.shape

        # train the final synthetic gradient to be close to 0
        if start+truncation_delta == seq_len:
            zeros = to_device(torch.zeros(real_grad_x_shape, requires_grad=False))
            loss = loss_function_synth(second_synth_grad, zeros)
            loss.backward()

        optimizer.step()

    with torch.no_grad():
        prediction, _, _ = model(batch_x)
        pred, gt = format_pred_and_gt(prediction, batch_y, model.single_output)
        loss = loss_function(pred, gt)

    return loss.item()

def train_bptt(model, optimizer, loss_function, batch_x, batch_y):
    prediction = model(batch_x)
    prediction, gt = format_pred_and_gt(prediction, batch_y, model.single_output)
    # Compute the loss, gradients, and update the parameters
    loss = loss_function(prediction, gt)
    
    # reset gradient
    model.zero_grad()
    
    loss.backward()

    optimizer.step()
    return loss.item()


def train(model, optimizer, generate_data, loss_function, train_func):
    # data generation is dependend on the training task
    train_X, train_Y = generate_data(config.TRAIN_SIZE, config.SEQ_LENGTH)
    training_results = []

    logging.info('----------------- Started Training -----------------')
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        total_loss = 0
        batch_num = 0

        for batch_x, batch_y in get_batched_data(train_X, train_Y):
            total_loss += train_func(model, optimizer, loss_function, batch_x, batch_y)
            batch_num += 1

        result = 'Epoch {} \t => Loss: {} [Batch-Time = {}s]'.format(epoch, total_loss / batch_num, round(time.time() - start_time, 2))
        logging.info(result)
        
        if epoch % 25 == 0:
            logging.info("Saved model")
            save_checkpoint(model, optimizer, epoch)