import torch

def stable_softmax(x):
    x = x - x.max(dim=-1, keepdim=True)[0]
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)