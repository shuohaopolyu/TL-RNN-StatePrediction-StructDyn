import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rmse_loss(pred, target):
    return (torch.sqrt(torch.mean((pred - target)**2))).cpu().detach().numpy()

def mae_loss(pred, target):
    return (torch.mean(torch.abs(pred - target))).cpu().detach().numpy()

def mse_loss(pred, target):
    return (torch.mean((pred - target)**2)).cpu().detach().numpy()

def compute_metrics(pred, target):
    rmse = rmse_loss(pred, target)
    mae = mae_loss(pred, target)
    mse = mse_loss(pred, target)
    return np.array([rmse, mae, mse])