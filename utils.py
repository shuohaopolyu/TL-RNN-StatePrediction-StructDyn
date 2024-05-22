import numpy as np
import torch
from scipy import signal
import numpy.linalg as LA
from excitations import PSDExcitationGenerator, BandPassPSD
import matplotlib.pyplot as plt
import scipy.io as sio

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
        G_yy = tru_w_acc[:, i].reshape(signal_mtx.shape[0], signal_mtx.shape[0])
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


def waveform_generator_1():
    # generate the voltage waveform for the experiment
    time = np.linspace(0, 25, 7501)
    # waveform 2: white noise from -2 to 2
    psd = BandPassPSD(a_v=1.0, f_1=10.0, f_2=100.0)
    force = PSDExcitationGenerator(psd, tmax=30, fmax=300)
    force_func = force()
    force_data = force_func(time)
    force_data = 0.9 * force_data / np.max(np.abs(force_data))
    force_data = np.round(force_data, 4)
    time_force = [
        ["RIGOL:DG8:CSV DATA FILE", ""],
        ["TYPE:Arb", ""],
        ["AMP:2.0 Vpp", ""],
        ["PERIOD:25.0 S", ""],
        ["DOTS:7500", ""],
        ["MODE:INSERT", ""],
        ["Sample Rate:300.0", ""],
        ["AWG N:0", ""],
        ["x", "y[V]"],
    ]
    for i in range(len(time)):
        if i != 0:
            time_force.append(["", force_data[i]])

    with open("./dataset/csb/waveform_2vpp.csv", "wb") as f:
        np.savetxt(f, time_force, delimiter=",", fmt="%s")
    plt.plot(time, force_data)
    plt.show()


def waveform_generator_2():
    # generate the voltage waveform for the experiment
    time = np.linspace(0, 5, 1501)
    # waveform 2: white noise from -2 to 2
    psd = BandPassPSD(a_v=1.0, f_1=10.0, f_2=100.0)
    force = PSDExcitationGenerator(psd, tmax=10, fmax=300)
    force_func = force()
    force_data = force_func(time)
    force_data = 0.9 * force_data / np.max(np.abs(force_data))
    force_data = np.round(force_data, 4)
    time_force = [
        ["RIGOL:DG8:CSV DATA FILE", ""],
        ["TYPE:Arb", ""],
        ["AMP:2.0 Vpp", ""],
        ["PERIOD:5.0 S", ""],
        ["DOTS:1500", ""],
        ["MODE:INSERT", ""],
        ["Sample Rate:300.0", ""],
        ["AWG N:0", ""],
        ["x", "y[V]"],
    ]
    for i in range(len(time)):
        if i != 0:
            time_force.append(["", force_data[i]])

    with open("./dataset/csb/waveform_2vpp_5s.csv", "wb") as f:
        np.savetxt(f, time_force, delimiter=",", fmt="%s")
    plt.plot(time, force_data)
    plt.show()


def process_fbg_data(dir="./dataset/csb/Peaks.20240508162346.txt"):
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
        time_data.append(convert_fbg_time_to_num(time_temp))
    fbg_data = np.array(fbg_data).astype(float)
    time_data = np.array(time_data).reshape(-1, 1)
    # print(data_temp)
    return time_data, fbg_data


def convert_num_to_time(num):
    import datetime

    time = datetime.timedelta(days=num)
    return time


def convert_fbg_time_to_num(time_string):
    # print(time_string)
    hour = time_string[9:11]
    minute = time_string[12:14]
    second = time_string[15:23]
    days = int(hour) / 24 + int(minute) / 1440 + float(second) / 86400
    return days


def convert_dewe_time_to_num(time_num):
    return time_num - 739380


def process_dewe_data(dir="./dataset/csb/test_1_amp_3.mat"):
    # import the dewe data

    data = sio.loadmat(dir)
    acc1 = data["Data1_AI_1_AI_1"]
    acc2 = data["Data1_AI_2_AI_2"]
    acc3 = data["Data1_AI_3_AI_3"]
    force1 = data["Data1_AI_4_AI_4"]
    disp1 = data["Data1_AI_5_AI_5"]
    disp1_ini = np.mean(disp1[:800])
    disp1 = disp1 - disp1_ini
    time_data = data["Data1_time_AI_1_AI_1"]
    time_list = []
    for i in range(time_data.shape[0]):
        time_list.append(str(convert_num_to_time(time_data[i, 0]))[13:])
        time_data[i, 0] = convert_dewe_time_to_num(time_data[i, 0])
    return time_data, time_list, acc1, acc2, acc3, force1, disp1


def combine_data():
    dewe_mat_paths = ["./dataset/csb/test_1_amp_3.mat"]
    fbg_paths = ["./dataset/csb/Peaks.20240508162346.txt"]
    data_num = 33000
    for i, dewe_mat in enumerate(dewe_mat_paths):
        dewe_time, dewe_time_list, acc1, acc2, acc3, force1, disp1 = process_dewe_data(
            dewe_mat
        )
        # print(dewe_time_list[0], dewe_time[0])
        ini_time = dewe_time[0]
        for j, fbg in enumerate(fbg_paths):
            fbg_time, fbg_data = process_fbg_data(fbg)
            if max(fbg_time) > ini_time:
                file_idx = j
                break
        fbg_time, fbg_data = process_fbg_data(fbg_paths[file_idx])
        time_deviation = fbg_time - ini_time
        for k, v in enumerate(time_deviation):
            if v > 0:
                temp_idx = k
                break
        # print(temp_idx)
        fbg_time = fbg_time[temp_idx - 100 : temp_idx + data_num + 100, :]
        fbg_data = fbg_data[temp_idx - 100 : temp_idx + data_num + 100, :]
        fbg_data_interp = np.zeros((data_num, 4))
        for q in range(4):
            fbg_data_interp[:, q] = np.interp(
                dewe_time[:data_num, 0], fbg_time[:, 0], fbg_data[:, q]
            )
        fbg_data_interp = np.round(fbg_data_interp, 4)
        fbg1 = fbg_data_interp[:, 0].reshape(-1, 1)
        fbg2 = fbg_data_interp[:, 2].reshape(-1, 1)
        fbg3 = fbg_data_interp[:, 3].reshape(-1, 1)
        # fbg4 = fbg_data_interp[:, 3].reshape(-1, 1)
        fbg1_ini = np.mean(fbg1[:800])
        fbg2_ini = np.mean(fbg2[:800])
        fbg3_ini = np.mean(fbg3[:800])
        # fbg4_ini = np.mean(fbg4[:800])
        mdict = {
            "acc1": acc1[:data_num, :],
            "acc2": acc2[:data_num, :],
            "acc3": acc3[:data_num, :],
            "force1": force1[:data_num, :],
            "disp1": disp1[:data_num, :],
            "fbg1": fbg1,
            "fbg2": fbg2,
            "fbg3": fbg3,
            # "fbg4": fbg4,
            "fbg1_ini": fbg1_ini,
            "fbg2_ini": fbg2_ini,
            "fbg3_ini": fbg3_ini,
            # "fbg4_ini": fbg4_ini,
        }
        sio.savemat("./dataset/csb/exp_{}.mat".format(i + 1), mdict)
