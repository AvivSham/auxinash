import torch
import torch.nn.functional as F
from torch import nn


class LogMSELoss(nn.Module):
    def __init__(self, tol=1e-10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.tol = tol

    def forward(self, pred, actual):
        return self.mse(
            torch.log(F.relu(pred) + self.tol), torch.log(F.relu(actual) + self.tol)
        )


class LogL1Loss(nn.Module):
    def __init__(self, tol=1e-10):
        super().__init__()
        self.loss = nn.L1Loss()
        self.tol = tol

    def forward(self, pred, actual):
        return self.loss(
            torch.log(F.relu(pred) + self.tol), torch.log(F.relu(actual) + self.tol)
        )


class LogNormLoss(nn.Module):
    def __init__(self, tol=1e-10):
        super().__init__()
        self.tol = tol

    def forward(self, pred, actual):
        return torch.norm(
            torch.log(F.relu(pred) + self.tol) - torch.log(F.relu(actual) + self.tol)
        )


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss
