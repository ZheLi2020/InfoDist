from __future__ import print_function

import torch
import torch.nn as nn


def get_logp_boundary(logps, mask, pos_beta=0.9, margin_tau=0.1):
    """
    Find the equivalent log-likelihood decision boundaries from normal log-likelihood distribution.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, )
        pos_beta: position hyperparameter: beta
        margin_tau: margin hyperparameter: tau
    """
    # calculate boundary for current class
    class_logps = logps[mask == 1].detach()
    n_idx = int(((mask == 1).sum() * pos_beta).item())
    sorted_indices = torch.sort(class_logps, descending=True)[1]

    n_idx = sorted_indices[n_idx-1]
    b_c = class_logps[n_idx]  # class boundary
    b_non = b_c - margin_tau  # non class boundary

    return b_c, b_non


def calculate_bg_spp_loss(logps, mask, boundaries, weights=None):
    """
    Calculate boudary guided semi-push-pull contrastive loss.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
    """
    # logps = logps / normalizer
    b_c = boundaries[0]  # class boundaries
    class_logps = logps[mask == 1]
    class_logps_inter = class_logps[class_logps <= b_c]
    loss_c = b_c - class_logps_inter

    b_non = boundaries[1]
    non_logps = logps[mask == 0]
    non_out = non_logps >= b_non
    loss_non = torch.tensor(0, dtype=torch.float)
    if non_out.sum() > 0:
        non_logps_inter = non_logps[non_out]
        loss_non = non_logps_inter - b_non

    if weights is not None:
        nor_weights = weights[mask == 1][class_logps <= b_c]
        loss_c = loss_c * nor_weights
        ano_weights = weights[mask == 0][class_logps >= b_non]
        loss_non = loss_non * ano_weights

    loss_c = torch.mean(loss_c)
    loss_non = torch.mean(loss_non)
    loss = loss_c + loss_non

    if loss_non.isnan():
        print('loss is nan')

    return loss


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def calculate_contrastive_loss(output, mask):
    class_logps = output[mask == 1]
    loss_c = torch.sum(euclidean_dist(class_logps, class_logps))/2
    loss_c = loss_c/class_logps.size(0)

    non_logps = output[mask == 0]

    loss_cross = torch.sum(euclidean_dist(class_logps, non_logps))/2
    loss_cross = loss_cross / non_logps.size(0)
    loss = loss_c - loss_cross

    return loss

