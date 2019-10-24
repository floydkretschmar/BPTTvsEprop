import torch
import torch.jit as jit

import config

def to_device(torch_object):
    if torch.cuda.is_available():
        return torch_object.cuda()
    else:
        return torch_object

def save_checkpoint(model, optimizer, amp, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp': amp.state_dict()
    }
    torch.save(checkpoint, '{}{}_{}.pt'.format(config.SAVE_PATH, model.get_name(), epoch))

def load_checkpoint(model, optimizer, amp):
    checkpoint = torch.load('{}{}.pt'.format(config.LOAD_PATH, model.get_name()))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])