import torch
import numpy as np


def gaussian_loss(prediction, target, mus, sigmas, null_val = np.nan):
    """Masked gaussian loss. Kindly note that the gaussian loss is calculated based on mu, sigma, and target. The prediction is sampled from N(mu, sigma), and is not used in the loss calculation (it will be used in the metrics calculation).

    Args:
        prediction (torch.Tensor): prediction of model. [B, L, N, 1].
        target (torch.Tensor): ground truth. [B, L, N, 1].
        mus (torch.Tensor): the mean of gaussian distribution. [B, L, N, 1].
        sigmas (torch.Tensor): the std of gaussian distribution. [B, L, N, 1]
        null_val (optional): null value. Defaults to np.nan.
    """
    # mask
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    distribution = torch.distributions.Normal(mus, sigmas)
    likelihood = distribution.log_prob(target)
    likelihood = likelihood * mask
    loss_g = -torch.mean(likelihood)
    return loss_g
