import torch as th


def huber(x, k=1.0):
    return th.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
