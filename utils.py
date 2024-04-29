import numpy as np
import torch
from scipy import signal
import numpy.linalg as LA
from excitations import PSDExcitationGenerator, BandPassPSD
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fdd(signal_mtx, f_lb=0.3, f_ub=0.8, nperseg_num=1000, fs=20):
    # implementation of frequency domain decomposition
    # this function is not used in the final version, I used pyoma2 instead.
    # signal should prepared in matrix form, whose dimension is ns*n_t
    # will return the mode shapes and the natural frequency
    """_summary_
    Args:
        signal_mtx (array): signal matrix, whose dimension is ns*n_t
        f_lb (float): lower bound of frequency. Defaults to 0.3.
        f_ub (float): upper bound of frequency. Defaults to 0.8.
        nperseg_num (int): number of data points in each segment. Defaults to 1000.
        fs (int): sampling frequency. Defaults to 20.

    Returns:
        ms_peak (array): mode shape
        nf (float): natural frequency
    """
    w_f = []
    w_acc = []
    for i in range(signal_mtx.shape[0]):
        for j in range(signal_mtx.shape[0]):
            w_f_temp, w_acc_temp = signal.csd(
                signal_mtx[i, :],
                signal_mtx[j, :],
                fs=fs,
                window="hann",
                nperseg=nperseg_num,
                axis=0,
                scaling="density",
                average="mean",
            )
            w_f.append(w_f_temp)
            w_acc.append(w_acc_temp)
    idx = [i for i, v in enumerate(w_f[0]) if v <= f_ub and v >= f_lb]
    tru_w_f = np.array(w_f)[0, idx]
    tru_w_acc = np.array(w_acc)[:, idx]
    sv = []
    ms = []
    for i in range(tru_w_acc.shape[1]):
        G_yy = tru_w_acc[:, i].reshape(
            signal_mtx.shape[0], signal_mtx.shape[0])
        u, s, _ = LA.svd(G_yy, full_matrices=True)
        sv.append(s[0])
        ms.append(np.real(u[:, 0]))
    nf_temp_idx = np.argmax(np.array(sv))
    nf_idx = idx[0] + nf_temp_idx
    nf = w_f[0][nf_idx]
    ms_peak = np.array(ms)[nf_temp_idx, :]
    return ms_peak, nf


def mac(pred, target):
    # pred and target are both mode shapes
    return np.abs(np.dot(pred, target)) ** 2 / (
        np.dot(pred, pred) * np.dot(target, target)
    )


def similarity(pred, target):
    # pred and target are 3d array, whose dimension is 10*800*26
    error_mtx = np.zeros((pred.shape[0], pred.shape[2]))
    for i in range(pred.shape[0]):
        for j in range(pred.shape[2]):
            pred_vec = pred[i, :, j]
            target_vec = target[i, :, j]
            mean_target = np.mean(target_vec)
            # error_mtx[i, j] = 1 - np.linalg.norm(
            #     pred_vec - target_vec
            # ) / np.linalg.norm(target_vec - mean_target)
            error_mtx[i, j] = np.linalg.norm(pred_vec - target_vec) / np.linalg.norm(
                target_vec - mean_target
            )
            # error_mtx[i, j] = np.linalg.norm(pred_vec - target_vec)
    return error_mtx


def waveform_generator():
    # generate the voltage waveform for the experiment
    # waveform 1: white noise from -5 to 5
    psd = BandPassPSD(a_v=1.0, f_1=10.0, f_2=410.0)
    force = PSDExcitationGenerator(psd, tmax=10, fmax=2000)
    time = np.linspace(0, 5, 5001)
    force_func = force()
    force_data = force_func(time)
    force_data = 4.9 * force_data / np.max(np.abs(force_data))
    force_data = np.round(force_data, 4)
    time_force = [
        ["RIGOL:DG8:CSV DATA FILE", ""],
        ["TYPE:Arb", ""],
        ["AMP:10.0000 Vpp", ""],
        ["PERIOD:50.00 S", ""],
        ["DOTS:5000", ""],
        ["MODE:INSERT", ""],
        ["Sample Rate:100.00", ""],
        ["AWG N:0", ""],
        ["x", "y[V]"],
    ]
    # aug_time_force = ["x", "y"] + [time_force]
    for i in range(len(time)):
        if i != 0:
            # time_force.append([time[i], force_data[i]])
            time_force.append(["", force_data[i]])

    with open("./dataset/csb/force_10.csv", "wb") as f:
        # np.savetxt(f, force_data, delimiter=",")
        np.savetxt(f, time_force, delimiter=",", fmt="%s")
    plt.plot(time, force_data)
    plt.show()
    # waveform 2: white noise from -2 to 2
    psd = BandPassPSD(a_v=1.0, f_1=10.0, f_2=410.0)
    force = PSDExcitationGenerator(psd, tmax=10, fmax=2000)
    force_func = force()
    force_data = force_func(time)
    force_data = 1.9 * force_data / np.max(np.abs(force_data))
    force_data = np.round(force_data, 4)
    time_force = [
        ["RIGOL:DG8:CSV DATA FILE", ""],
        ["TYPE:Arb", ""],
        ["AMP:4.0 Vpp", ""],
        ["PERIOD:50.0 S", ""],
        ["DOTS:5000", ""],
        ["MODE:INSERT", ""],
        ["Sample Rate:100.0", ""],
        ["AWG N:0", ""],
        ["x", "y[V]"],
    ]
    for i in range(len(time)):
        if i != 0:
            # time_force.append([time[i], force_data[i]])
            time_force.append(["", force_data[i]])

    with open("./dataset/csb/force_4.csv", "wb") as f:
        np.savetxt(f, time_force, delimiter=",", fmt="%s")
    plt.plot(time, force_data)
    plt.show()


def import_fbg_data(dir="./dataset/experiment/fbg/noise.txt"):
    # import the fbg data
    with open(dir, "r") as f:
        data = f.readlines()
    fbg_data = []
    time_data = []
    for i in range(45, len(data)):
        temp = data[i].split("	")
        time_temp = temp[0]
        data_temp = temp[-4:]
        data_temp[-1] = data_temp[-1].replace("\n", "")
        fbg_data.append(data_temp)
        time_data.append(time_temp)
    fbg_data = np.array(fbg_data).astype(float)
    plt.plot(fbg_data[:, 0])
    plt.plot(fbg_data[:, 1])
    plt.plot(fbg_data[:, 2])
    plt.plot(fbg_data[:, 3])
    plt.show()
    return time_data, fbg_data
