import math
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from functools import partial


def mse(output, target):
    return F.mse_loss(output, target)


def mse_with_context(output, target, context_len):
    return F.mse_loss(output[:, context_len:], target[:, context_len:])


def log_mse(output, target):
    return torch.log(F.mse_loss(output, target))


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


def cross_entropy_with_context(output, target, context_len):
    output = output[:, context_len:]
    target = target[:, context_len:]
    output = output.reshape(-1, output.shape[-1])
    target = target.reshape(-1)
    return F.cross_entropy(output, target)


def phase_pick_loss(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    h = y_true * torch.log(F.softmax(y_pred, dim=-1) + eps)
    h = h.mean(1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h


def bidirAutoregLoss(x, y):
    x_ntk, x_ptk, x_tokens = x
    if x_tokens is None:
        diff = int(x_ntk.shape[1] - y.shape[1])
        next_token_loss = F.mse_loss(x_ntk[:, :-diff, :], y)
        prev_token_loss = F.mse_loss(x_ptk[:, diff:, :], y)
    else:
        diff = int(x_ntk.shape[1] - x_tokens.shape[1])
        next_token_loss = F.mse_loss(x_ntk[:, :-diff, :], x_tokens)
        prev_token_loss = F.mse_loss(x_ptk[:, diff:, :], x_tokens)
    return next_token_loss + prev_token_loss



metric_functions = {
    'mse': mse,
    'log_mse': log_mse,
    'mse-context': mse_with_context,
    'mae': mae,
    'accuracy': accuracy,
    'cross-entropy': cross_entropy,
    'cross-entropy-context': cross_entropy_with_context,
    'phase-pick': phase_pick_loss,
    'bidir-autoreg-loss': bidirAutoregLoss,
}
