import torch
from torch.nn import functional as F


def greedy_prediction(pred: torch.Tensor):
    """
    Greedy prediction. Argmax(pred)
    :param pred: predictions [1, num_classes]
    :return: greedy prediction [1, 1]
    """
    return torch.argmax(pred, dim=-1).unsqueeze(0)


def multinomial_prediction(pred: torch.Tensor, temperature: float = 1.0):
    """
    Sample from predictions using softmax and multinomial distribution.
    :param pred: predictions [1, num_classes]
    :param temperature: temperature for softmax. Default: 1.0
    :return: sampled prediction [1, 1]
    """
    pred = pred / temperature
    return torch.multinomial(F.softmax(pred, dim=1), 1)


def top_k_prediction(pred: torch.Tensor, k: int, temperature: float = 1.0):
    """
    Sample from top k predictions
    :param pred: prediction before softmax [1, num_classes]
    :param k: number of classes to sample from
    :param temperature: temperature for softmax. Default: 1.0
    :return: sampled prediction [1, 1]
    """
    pred = pred / temperature
    pred = F.softmax(pred, dim=-1)
    top_k_prob, top_k_idx = torch.topk(pred, k)
    sample = torch.multinomial(top_k_prob, num_samples=1)
    return top_k_idx[:, int(sample)].unsqueeze(0)
