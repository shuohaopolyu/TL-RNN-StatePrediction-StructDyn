import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
import  numpy.linalg as LA
# set the fonttype to be Arial
plt.rcParams["font.family"] = "Arial"
# set the font size's default value
plt.rcParams.update({"font.size": 12})


def acceleration_measurement():
    # load the seismic response
    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    # plot the seismic response
    time = solution["time"]
    acc = solution["acc"]
    # four subfigures to plot the time history of the seismic response
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    ax1, ax2, ax3, ax4 = axs.flatten()

    ax1.plot(time, acc[0, :], label="1st floor")
    ax1.set_ylabel("Acceleration (m/s^2)")

    ax2.plot(time, acc[4, :], label="5nd floor")

    ax3.plot(time, acc[8, :], label="9rd floor")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Acceleration (m/s^2)")

    ax4.plot(time, acc[12, :], label="13th floor")
    ax4.set_xlabel("Time (s)")

    plt.show()

def psd_acc():
    # load the seismic response
    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    # plot the seismic response
    time = solution["time"]
    acc = solution["acc"]
    acc_test = np.squeeze(acc[12, :])
    N = len(acc_test)
    yf_test = np.fft.fft(acc_test)
    xf = np.linspace(0.0, 10, N // 2)
    # plot the PSD of the acceleration
    f, Pxx = signal.welch(acc, fs=20,  window="hann", nperseg=1000, noverlap=500, axis=1)
    # plt.plot(xf, 2.0 * time[-1] * (np.abs(yf_test[: N // 2])**2)/N**2)
    plt.plot(f, Pxx.T)
    plt.yscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.show()

def fdd(signal_mtx, f_lb=0.3, f_ub=0.8, nperseg_num=1000, type='peak'):
    # implementation of frequency domain decomposition
    # signal should in matrix form, whose dimension is 5*n_t
    # will return the mode shapes and the natural frequency
    # two ways to generate mode shapes, peak or average
    w_f = []
    w_acc = []
    for i in range(signal_mtx.shape[0]):
        for j in range(signal_mtx.shape[0]):
            w_f_temp, w_acc_temp = signal.csd(
                signal_mtx[i, :], signal_mtx[j, :], fs=20, window='hann', nperseg=nperseg_num, axis=0, scaling='density', average='mean')
            w_f.append(w_f_temp)
            w_acc.append(w_acc_temp)
    idx = [i for i, v in enumerate(w_f[0]) if v <= f_ub and v >= f_lb]
    tru_w_acc = np.array(w_acc)[:, idx]
    nf_temp_idx = []
    ms = []
    for i in range(tru_w_acc.shape[1]):
        G_yy = tru_w_acc[:, i].reshape(13, 13)
        u, s, _ = LA.svd(G_yy, full_matrices=True)
        nf_temp_idx.append(s[0])
        ms.append(np.real(u[:, 0]))
    nf_temp_idx = np.argmax(np.array(nf_temp_idx))
    nf_idx = idx[0]+nf_temp_idx
    nf = w_f[0][nf_idx]
    if type == 'peak':
        ms_peak = np.array(ms)[nf_temp_idx, :]
        return ms_peak, nf
    elif type == 'average':
        ms_avg = np.average(np.array(ms), axis=0)
        return ms_avg, nf

def ms_acc(f_lb=1.1, f_ub=1.5):
    # load the seismic response
    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    # plot the seismic response
    acc_mtx = solution["acc"]
    ms, nf = fdd(acc_mtx, f_lb=f_lb, f_ub=f_ub, nperseg_num=1000, type='peak')
    plt.plot(range(13), ms, '-o')
    plt.xlabel('DOF')
    plt.ylabel('Mode Shape')

def mode_shape():
    # load the seismic response
    data_path = "./dataset/bists/ambient_response.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    acc_mtx = solution["acc"]
    f_lb_list = [0.3, 1.3, 2.2, 3.0, 3.8, 4.5]
    f_ub_list = [0.8, 1.8, 2.7, 3.6, 4.4, 5.1]
    fig, axs = plt.subplots(1, 6, figsize=(12, 6))
    for i in range(6):
        ms, nf = fdd(acc_mtx, f_lb=f_lb_list[i], f_ub=f_ub_list[i], nperseg_num=1000, type='peak')
        ms = ms / np.max(np.abs(ms))
        axs[i].plot(ms, range(13), '-s', label='f_lb='+str(f_lb_list[i])+'f_ub='+str(f_ub_list[i]), color='blue', markersize=4, linewidth=1.5)
        axs[i].tick_params(axis='both', direction='in', labelsize=8)
        axs[i].set_ylim(-0.5, 12.5)
        axs[i].set_xlim(-1.1, 1.1)
        axs[i].grid(True)
        axs[i].set_yticks(range(13), [str(i+1) for i in range(13)])
        if i == 0:
            axs[i].set_ylabel('DOF')

    plt.show()


