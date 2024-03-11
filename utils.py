import numpy as np
import torch
from scipy import signal
import numpy.linalg as LA

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

def fdd(signal_mtx, f_lb=0.3, f_ub=0.8, nperseg_num=1000, fs=20, psd=False):
    # implementation of frequency domain decomposition
    # signal should in matrix form, whose dimension is ns*n_t
    # will return the mode shapes and the natural frequency
    w_f = []
    w_acc = []
    for i in range(signal_mtx.shape[0]):
        for j in range(signal_mtx.shape[0]):
            w_f_temp, w_acc_temp = signal.csd(
                signal_mtx[i, :], signal_mtx[j, :], fs=fs, window='hann', nperseg=nperseg_num, axis=0, scaling='density', average='mean')
            w_f.append(w_f_temp)
            w_acc.append(w_acc_temp)
    idx = [i for i, v in enumerate(w_f[0]) if v <= f_ub and v >= f_lb]
    tru_w_f = np.array(w_f)[0, idx]
    tru_w_acc = np.array(w_acc)[:, idx]
    sv = []
    ms = []
    for i in range(tru_w_acc.shape[1]):
        G_yy = tru_w_acc[:, i].reshape(signal_mtx.shape[0], signal_mtx.shape[0])
        u, s, _ = LA.svd(G_yy, full_matrices=True)
        sv.append(s[0])
        ms.append(np.real(u[:, 0]))
    nf_temp_idx = np.argmax(np.array(sv))
    nf_idx = idx[0]+nf_temp_idx
    nf = w_f[0][nf_idx]
    ms_peak = np.array(ms)[nf_temp_idx, :]
    if not psd:
        return ms_peak, nf
    else:
        return tru_w_f, np.array(sv)


def mac(pred, target):
    # pred and target are both mode shapes
    return np.abs(np.dot(pred, target))**2 / (np.dot(pred, pred) * np.dot(target, target))

def self_psd(f_n, zeta, q, f_test):
    h = ((1-(f_test/f_n)**2)/((1-(f_test/f_n)**2)**2+(2*zeta*f_test/f_n)**2) - 1j*2*zeta*f_test/f_n/((1-(f_test/f_n)**2)**2+(2*zeta*f_test/f_n)**2)) * q
    return np.abs(h)**2

def similarity(pred, target):
    # pred and target are 3d array, whose dimension is 10*800*26
    error_mtx = np.zeros((pred.shape[0], pred.shape[2]))
    for i in range(pred.shape[0]):
        for j in range(pred.shape[2]):
            pred_vec = pred[i, :, j]
            target_vec = target[i, :, j]
            mean_target = np.mean(target_vec)
            error_mtx[i, j] = 1- np.linalg.norm(pred_vec - target_vec) / np.linalg.norm(target_vec - mean_target)
    return error_mtx
    
