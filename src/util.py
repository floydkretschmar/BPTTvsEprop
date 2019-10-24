import torch
import torch.jit as jit

def to_device(torch_object):
    if torch.cuda.is_available():
        return torch_object.cuda()
    else:
        return torch_object