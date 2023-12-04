import torch
import numpy as np


def gaussian_loss(prediction, real_value, mu, sigma, null_val = np.nan):
    """Masked gaussian loss. Kindly note that the gaussian loss is calculated based on mu, sigma, and real_value. The prediction is sampled from N(mu, sigma), and is not used in the loss calculation (it will be used in the metrics calculation).

    Args:
        prediction (torch.Tensor): prediction of model. [B, L, N, 1].
        real_value (torch.Tensor): ground truth. [B, L, N, 1].
        mu (torch.Tensor): the mean of gaussian distribution. [B, L, N, 1].
        sigma (torch.Tensor): the std of gaussian distribution. [B, L, N, 1]
        null_val (optional): null value. Defaults to np.nan.
    """
    # mask
    if np.isnan(null_val):
        mask = ~torch.isnan(real_value)
    else:
        eps = 5e-5
        mask = ~torch.isclose(real_value, torch.tensor(null_val).expand_as(real_value).to(real_value.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    distribution = torch.distributions.Normal(mu, sigma)
    likelihood = distribution.log_prob(real_value)
    likelihood = likelihood * mask
    loss_g = -torch.mean(likelihood)
    return loss_g
