import numpy as np
import torch
import torch.nn.functional as F

def discriminator_loss(d_same, d_diff):
    """
    Aim to maximize the probability that the discriminator
    predict correctly whether two pose vectors are from the
    same video or not
    """
    # Define target of the discriminator
    # if the pose vectors are of the same video, target should be 1
    target_same = torch.ones(d_same.shape[0], 1, dtype=torch.float)

    # if the pose vectors are of different videos, target should be 0
    target_diff = torch.zeros(d_same.shape[0], 1, dtype=torch.float)


    loss_same = F.binary_cross_entropy(d_same, target_same)
    loss_diff = F.binary_cross_entropy(d_diff, target_diff)

    # acc = (d_same.gt(0.5).mean() + d_diff.lt(0.5).mean()) / 2
    acc = None

    return loss_same + loss_diff, acc


def similarity_loss(ci_t, ci_tk):
    return F.mse_loss(ci_t, ci_tk)


def reconstruction_loss(predxi_tk, xi_tk):
    return F.mse_loss(xi_tk, predxi_tk)


def adversarial_loss(d_same, d_diff):
    """
    First half of the adversarial framework tries to classify the pair
    of pose vectors as being from the same/different video

    Second half of the framework tries to maximize the uncertainty (entropy)
    of the discriminator output on pairs of frame from the same clip
    """

    # First half loss
    loss_c, _ = discriminator_loss(d_same, d_diff)
    
    # Second half loss
    target_ep = torch.ones(d_same.shape[0], 1, dtype=torch.float) / 2
    loss_ep = F.binary_cross_entropy(d_same, target_ep)

    return loss_c + loss_ep

