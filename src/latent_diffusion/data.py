"""Reproducible held-out digit-pair addition task."""
import random
import torch

def addition_data(seed=17, train_fraction=.8):
    pairs = [(a, b) for a in range(10) for b in range(10)]
    random.Random(seed).shuffle(pairs)
    prompts = torch.tensor(pairs, dtype=torch.long)
    summed = prompts.sum(1)
    target = torch.stack((summed // 10, summed % 10), dim=-1)
    n = int(len(pairs) * train_fraction)
    return (prompts[:n], target[:n]), (prompts[n:], target[n:])
