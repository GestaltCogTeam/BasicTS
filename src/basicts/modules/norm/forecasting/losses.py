"""Training losses from the official APT normalization pipeline."""

import torch
from torch.nn import functional as F


def apt_regularization(module, weight, bias):
    embeddings = module.get_combined_embeddings()
    orthogonal = weight.new_zeros(())
    for embedding in embeddings:
        similarity = embedding @ embedding.T
        size = len(embedding)
        diagonal = (1 - similarity.diag()).square().mean()
        off_diagonal = similarity[~torch.eye(size, dtype=torch.bool, device=embedding.device)]
        orthogonal = orthogonal + diagonal + (off_diagonal.square().mean() if size > 1 else 0)
    orthogonal = orthogonal / len(embeddings)
    load = module.get_load() + 1e-8
    probability = load / load.sum()
    balance = (probability - 1 / load.numel()).square().sum()
    def affine_penalty(value):
        # The reference unbiased variance is undefined for a singleton batch.
        flat = value.reshape(-1)
        variance = flat.var(unbiased=flat.numel() > 1)
        return (1 - (variance + 1e-4).sqrt()).square() + flat.mean().square()
    return orthogonal + balance + affine_penalty(weight) + affine_penalty(bias)


def station_loss(targets, statistics, period_len):
    blocks = targets.reshape(targets.size(0), -1, period_len, targets.size(2))
    actual = torch.cat((blocks.mean(2), blocks.std(2)), dim=-1)
    return F.mse_loss(statistics, actual)
