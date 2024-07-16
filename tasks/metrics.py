import math
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from functools import partial


def mse(output, target):
    return F.mse_loss(output, target)


def mae(output, target):
    return F.l1_loss(output, target)


def accuracy(output, target):
    output = output.view(-1, output.shape[-1])
    if target.numel() > output.shape[0]:
        # Mixup leads to this case: use argmax class
        target = target.argmax(dim=-1)
    target = target.view(-1)
    return torch.eq(torch.argmax(output, dim=-1), target).float().mean()


def cross_entropy(output, target):
    output = output.view(-1, output.shape[-1])
    target = target.view(-1)
    return F.cross_entropy(output, target)


metric_functions = {
    'mse': mse,
    'mae': mae,
    'accuracy': accuracy,
    'cross-entropy': cross_entropy
}
