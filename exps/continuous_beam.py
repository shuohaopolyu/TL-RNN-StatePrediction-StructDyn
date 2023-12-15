import numpy as np
import pickle
from systems import ContinuousBridge
from excitations import PSDExcitationGenerator, RollOffPSD, FlatNoisePSD
import torch
from models import AutoEncoder, Rnn, Lstm
import os
from utils import compute_metrics
from numpy import linalg as LA


def compute_response(excitation_pattern=1):
    # Define the force
    if excitation_pattern == 1:
        psd = FlatNoisePSD(a_v=1e8)
    elif excitation_pattern == 2:
        psd = RollOffPSD(a_v=1e12, omega_r=50.0, omega_c=80.0)
    force = PSDExcitationGenerator(psd=psd, tmax=100, fmax=50)
    force_fun_1, force_fun_2, force_fun_3, force_fun_4 = (
        force(),
        force(),
        force(),
        force(),
    )

    # Define the system
    cb = ContinuousBridge(
        rho=1e4,
        E=6e9,
        A=1,
        I=1 / 12,
        L=0.5,
        damp_params=(0, 5, 0.03),
        f_dof=[13, 30, 44, 69],
        resp_dof="full",
        t_eval=np.arange(0, 100, 1 / 100),
        f_t=[force_fun_1, force_fun_2, force_fun_3, force_fun_4],
        init_cond=np.zeros(172),
    )

    # generate the data
    full_data = cb.response(type="full", method="Radau")
    full_data["displacement"] = full_data["displacement"].T
    full_data["velocity"] = full_data["velocity"].T
    full_data["acceleration"] = full_data["acceleration"].T
    full_data["force"] = cb.f_mtx().T
    full_data["time"] = cb.t_eval.reshape(-1, 1)

    # save the data
    with open(
        f"./dataset/full_response_excitation_pattern_{excitation_pattern}.pkl", "wb"
    ) as f:
        pickle.dump(full_data, f)


def _train_an_ae(
    train_set, test_set_1, test_set_2, num_hid_neuron, epochs, learning_rate
):
    ae = AutoEncoder([86, 40, 40, num_hid_neuron])
    ae.train_AE(train_set, test_set_1, epochs, learning_rate, None, None)
    ae.eval()
    with torch.no_grad():
        pred1 = ae(train_set)
        pred2 = ae(test_set_1)
        pred3 = ae(test_set_2)

    metric_1 = compute_metrics(pred1, train_set)
    metric_2 = compute_metrics(pred2, test_set_1)
    metric_3 = compute_metrics(pred3, test_set_2)
    return metric_1, metric_2, metric_3


def _rand_generate_train_test_set(data_1, data_2, num_train, num_test):
    perm_1 = torch.randperm(data_1.shape[0])
    perm_2 = torch.randperm(data_2.shape[0])
    data_1 = data_1[perm_1, :]
    data_2 = data_2[perm_2, :]
    train_set = data_1[:num_train, :]
    test_set_1 = data_1[num_train : num_train + num_test, :]
    test_set_2 = data_2[:num_test, :]
    return train_set, test_set_1, test_set_2


def ae_num_hid_neuron():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
        full_data_1 = pickle.load(f)
    with open("./dataset/full_response_excitation_pattern_2.pkl", "rb") as f:
        full_data_2 = pickle.load(f)
    disp_1 = torch.tensor(full_data_1["displacement"], dtype=torch.float32).to(device)
    velo_1 = torch.tensor(full_data_1["velocity"], dtype=torch.float32).to(device)
    disp_2 = torch.tensor(full_data_2["displacement"], dtype=torch.float32).to(device)
    velo_2 = torch.tensor(full_data_2["velocity"], dtype=torch.float32).to(device)

    train_set_disp_1 = disp_1[:2000, :]
    test_set_disp_1 = disp_1[2000:10000, :]
    test_set_disp_2 = disp_2
    train_set_velo_1 = velo_1[:2000, :]
    test_set_velo_1 = velo_1[2000:10000, :]
    test_set_velo_2 = velo_2

    num_iter = 15
    errors = [np.zeros((num_iter, 3)) for _ in range(6)]

    (
        error_disp_train,
        error_disp_test_1,
        error_disp_test_2,
        error_velo_train,
        error_velo_test_1,
        error_velo_test_2,
    ) = errors

    for i in range(1, num_iter + 1):
        metrics_disp = _train_an_ae(
            train_set_disp_1, test_set_disp_1, test_set_disp_2, i, 50000, 6e-5
        )
        metrics_velo = _train_an_ae(
            train_set_velo_1, test_set_velo_1, test_set_velo_2, i, 50000, 1e-4
        )

        # Compute and append all metrics at once
        (
            error_disp_train[i - 1, :],
            error_disp_test_1[i - 1, :],
            error_disp_test_2[i - 1, :],
        ) = (
            metrics_disp[0],
            metrics_disp[1],
            metrics_disp[2],
        )
        (
            error_velo_train[i - 1, :],
            error_velo_test_1[i - 1, :],
            error_velo_test_2[i - 1, :],
        ) = (
            metrics_velo[0],
            metrics_velo[1],
            metrics_velo[2],
        )

    with open("./dataset/ae_num_hid_neuron.pkl", "wb") as f:
        pickle.dump(
            {
                "error_disp_train": error_disp_train,
                "error_disp_test_1": error_disp_test_1,
                "error_disp_test_2": error_disp_test_2,
                "error_velo_train": error_velo_train,
                "error_velo_test_1": error_velo_test_1,
                "error_velo_test_2": error_velo_test_2,
            },
            f,
        )


def ae_train_data_size():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
        full_data_1 = pickle.load(f)
    with open("./dataset/full_response_excitation_pattern_2.pkl", "rb") as f:
        full_data_2 = pickle.load(f)
    disp_1 = torch.tensor(full_data_1["displacement"], dtype=torch.float32).to(device)
    velo_1 = torch.tensor(full_data_1["velocity"], dtype=torch.float32).to(device)
    disp_2 = torch.tensor(full_data_2["displacement"], dtype=torch.float32).to(device)
    velo_2 = torch.tensor(full_data_2["velocity"], dtype=torch.float32).to(device)
    num_data = [20, 50, 100, 200, 500, 1000, 2000, 4000, 8000]
    num_trials = 10
    for i, num in enumerate(num_data):
        errors = [np.zeros((num_trials, 3)) for _ in range(6)]
        (
            error_disp_train,
            error_disp_test_1,
            error_disp_test_2,
            error_velo_train,
            error_velo_test_1,
            error_velo_test_2,
        ) = errors
        errors_mean = [np.zeros((len(num_data), 3)) for _ in range(6)]
        (
            error_disp_train_mean,
            error_disp_test_1_mean,
            error_disp_test_2_mean,
            error_velo_train_mean,
            error_velo_test_1_mean,
            error_velo_test_2_mean,
        ) = errors_mean
        errors_std = [np.zeros((len(num_data), 3)) for _ in range(6)]
        (
            error_disp_train_std,
            error_disp_test_1_std,
            error_disp_test_2_std,
            error_velo_train_std,
            error_velo_test_1_std,
            error_velo_test_2_std,
        ) = errors_std
        for j in range(num_trials):
            (
                train_set_disp_1,
                test_set_disp_1,
                test_set_disp_2,
            ) = _rand_generate_train_test_set(disp_1, disp_2, num, 2000)
            (
                train_set_velo_1,
                test_set_velo_1,
                test_set_velo_2,
            ) = _rand_generate_train_test_set(velo_1, velo_2, num, 2000)
            metrics_disp = _train_an_ae(
                train_set_disp_1, test_set_disp_1, test_set_disp_2, 4, 50000, 1e-5
            )
            metrics_velo = _train_an_ae(
                train_set_velo_1, test_set_velo_1, test_set_velo_2, 8, 50000, 1e-4
            )
            error_disp_train[j, :], error_disp_test_1[j, :], error_disp_test_2[j, :] = (
                metrics_disp[0],
                metrics_disp[1],
                metrics_disp[2],
            )
            error_velo_train[j, :], error_velo_test_1[j, :], error_velo_test_2[j, :] = (
                metrics_velo[0],
                metrics_velo[1],
                metrics_velo[2],
            )
        (
            error_disp_train_mean[i, :],
            error_disp_test_1_mean[i, :],
            error_disp_test_2_mean[i, :],
        ) = (
            np.mean(error_disp_train, axis=0),
            np.mean(error_disp_test_1, axis=0),
            np.mean(error_disp_test_2, axis=0),
        )
        (
            error_velo_train_mean[i, :],
            error_velo_test_1_mean[i, :],
            error_velo_test_2_mean[i, :],
        ) = (
            np.mean(error_velo_train, axis=0),
            np.mean(error_velo_test_1, axis=0),
            np.mean(error_velo_test_2, axis=0),
        )
        (
            error_disp_train_std[i, :],
            error_disp_test_1_std[i, :],
            error_disp_test_2_std[i, :],
        ) = (
            np.std(error_disp_train, axis=0),
            np.std(error_disp_test_1, axis=0),
            np.std(error_disp_test_2, axis=0),
        )
        (
            error_velo_train_std[i, :],
            error_velo_test_1_std[i, :],
            error_velo_test_2_std[i, :],
        ) = (
            np.std(error_velo_train, axis=0),
            np.std(error_velo_test_1, axis=0),
            np.std(error_velo_test_2, axis=0),
        )
    with open("./dataset/ae_train_data_size.pkl", "wb") as f:
        pickle.dump(
            {
                "error_disp_train_mean": error_disp_train_mean,
                "error_disp_test_1_mean": error_disp_test_1_mean,
                "error_disp_test_2_mean": error_disp_test_2_mean,
                "error_velo_train_mean": error_velo_train_mean,
                "error_velo_test_1_mean": error_velo_test_1_mean,
                "error_velo_test_2_mean": error_velo_test_2_mean,
                "error_disp_train_std": error_disp_train_std,
                "error_disp_test_1_std": error_disp_test_1_std,
                "error_disp_test_2_std": error_disp_test_2_std,
                "error_velo_train_std": error_velo_train_std,
                "error_velo_test_1_std": error_velo_test_1_std,
                "error_velo_test_2_std": error_velo_test_2_std,
            },
            f,
        )


def ae_full_state_vs_sep_disp_velo():
    pass


def _dkf(
    starts,
    steps,
    sensor_idx,
    num_modes,
    excitation_pattern,
    dkf_params=[1e-20, 1e-2, 3e9],
    output="displacement",
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if excitation_pattern == 1:
        with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
            full_data = pickle.load(f)
    elif excitation_pattern == 2:
        with open("./dataset/full_response_excitation_pattern_2.pkl", "rb") as f:
            full_data = pickle.load(f)
    acc = torch.tensor(full_data["acceleration"], dtype=torch.float32).to(device).T
    obs_data = acc[sensor_idx, starts : steps + starts]

    # Define the system
    cb = ContinuousBridge(
        rho=1e4,
        E=6e9,
        A=1,
        I=1 / 12,
        L=0.5,
        damp_params=(0, 5, 0.03),
        f_dof=[13, 30, 44, 69],
        resp_dof="full",
        t_eval=np.arange(0, 100, 1 / 100),
        f_t=None,
        init_cond=np.zeros(172),
    )
    _, modes = cb.freqs_modes(mode_normalize=True)
    md_mtx = modes[:, 0:num_modes]

    # compute the state space matrices
    A, B, G, J = cb.truncated_state_space_mtx(truncation=num_modes, type="discrete")
    Q_zeta = np.eye(num_modes * 2) * dkf_params[0]
    R = np.eye(len(sensor_idx)) * dkf_params[1]
    Q_p = np.eye(4) * dkf_params[2]

    # initialization
    load_mtx = np.zeros((4, steps))
    zeta_mtx = np.zeros((num_modes * 2, steps))
    P_p_mtx = np.zeros((4, 4, steps))
    P_mtx = np.zeros((num_modes * 2, num_modes * 2, steps))
    P_p_mtx[:, :, 0] = Q_p
    P_mtx[:, :, 0] = Q_zeta

    # kalman filter for input and state estimation
    for i in range(steps - 1):
        # Prediction stage for the input:
        # Evolution of the input and prediction of covariance input:
        p_k_m = load_mtx[:, i]
        P_k_pm = P_p_mtx[:, :, i] + Q_p

        # Update stage for the input:
        # Calculation of Kalman gain for input:
        G_k_p = P_k_pm @ J.T @ LA.inv(J @ P_k_pm @ J.T + R)

        # Improve predictions of input using latest observation:
        load_mtx[:, i + 1] = p_k_m + G_k_p @ (
            obs_data[:, i + 1] - G @ zeta_mtx[:, i] - J @ p_k_m
        )
        P_p_mtx[:, :, i + 1] = P_k_pm - G_k_p @ J @ P_k_pm

        # Prediction stage for the state:
        # Evolution of state and prediction of covariance of state:
        zeta_k_m = A @ zeta_mtx[:, i] + B @ load_mtx[:, i + 1]
        P_k_m = A @ P_mtx[:, :, i] @ A.T + Q_zeta

        # Update stage for the state:
        # Calculation of Kalman gain for state:
        G_k_zeta = P_k_m @ G.T @ LA.inv(G @ P_k_m @ G.T + R)

        # Improve predictions of state using latest observation:
        zeta_mtx[:, i + 1] = zeta_k_m + G_k_zeta @ (
            obs_data[:, i + 1] - G @ zeta_k_m - J @ load_mtx[:, i + 1]
        )
        P_mtx[:, :, i + 1] = P_k_m - G_k_zeta @ G @ P_k_m

        progress_percentage = (i + 2) / (steps) * 100
        if progress_percentage % 10 == 0:
            print(f"Progress: {progress_percentage:.0f}%")
    if output == "displacement":
        disp_pred = md_mtx @ zeta_mtx[:num_modes, :]
        return disp_pred
    elif output == "velocity":
        vel_pred = md_mtx @ zeta_mtx[num_modes:, :]
        return vel_pred
    
def tune_dkf_params(start, steps):
    R_values = np.logspace(-4, 2, 100)
    disp_rmse = np.zeros(len(R_values))
    vel_rmse = np.zeros(len(R_values))
    with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
        full_data = pickle.load(f)
    disp = full_data["displacement"].T
    velo = full_data["velocity"].T
    for i, R_value in enumerate(R_values):
        disp_pred = _dkf(start, steps, [13, 17, 32, 50, 67, 77], 6, 1, [1e-20, R_value, 3e9], "displacement")
        vel_pred = _dkf(start, steps, [13, 17, 32, 50, 67, 77], 6, 1, [1e-20, R_value, 3e9], "velocity")
        disp_rmse[i] = np.sqrt(np.mean((disp_pred - disp)**2))
        vel_rmse[i] = np.sqrt(np.mean((vel_pred - velo)**2))
    results = {"R_values": R_values, "disp_rmse": disp_rmse, "vel_rmse": vel_rmse}
    with open("./dataset/dkf_tune.pkl", "wb") as f:
        pickle.dump(results, f)

def dkf_eval():
    dkf_disp_1 = _dkf(
        2000, 8000, [13, 17, 32, 50, 67, 77], 6, 1, [1e-20, 1e-2, 3e9], "displacement"
    )
    dkf_velo_1 = _dkf(
        2000, 8000, [13, 17, 32, 50, 67, 77], 6, 1, [1e-20, 2.66, 3e9], "velocity"
    )
    dkf_disp_2 = _dkf(
        0, 10000, [13, 17, 32, 50, 67, 77], 6, 2, [1e-20, 1e-2, 3e9], "displacement"
    )
    dkf_velo_2 = _dkf(
        0, 10000, [13, 17, 32, 50, 67, 77], 6, 2, [1e-20, 2.66, 3e9], "velocity"
    )

def _rnn():
    pass

def _lstm():
    pass

def _birnn():
    pass

def _bilstm():
    pass

def rnn_ae_performance_eval():
    pass


def rnn_ae_noise_robustness():
    pass


def rnn_ae_num_sensors():
    pass
