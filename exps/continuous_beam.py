import numpy as np
import pickle
from systems import ContinuousBridge
from excitations import PSDExcitationGenerator, RollOffPSD, FlatNoisePSD
import torch
from models import AutoEncoder, Rnn, Lstm, Mlp
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
    errors_mean = [np.zeros((len(num_data), 3)) for _ in range(6)]
    errors_max = [np.zeros((len(num_data), 3)) for _ in range(6)]
    errors_min = [np.zeros((len(num_data), 3)) for _ in range(6)]
    errors_std = [np.zeros((len(num_data), 3)) for _ in range(6)]
    (
        error_disp_train_mean,
        error_disp_test_1_mean,
        error_disp_test_2_mean,
        error_velo_train_mean,
        error_velo_test_1_mean,
        error_velo_test_2_mean,
    ) = errors_mean
    (
        error_disp_train_max,
        error_disp_test_1_max,
        error_disp_test_2_max,
        error_velo_train_max,
        error_velo_test_1_max,
        error_velo_test_2_max,
    ) = errors_max
    (
        error_disp_train_min,
        error_disp_test_1_min,
        error_disp_test_2_min,
        error_velo_train_min,
        error_velo_test_1_min,
        error_velo_test_2_min,
    ) = errors_min
    (
        error_disp_train_std,
        error_disp_test_1_std,
        error_disp_test_2_std,
        error_velo_train_std,
        error_velo_test_1_std,
        error_velo_test_2_std,
    ) = errors_std
    for i, num in enumerate(num_data):
        errors_trial = [np.zeros((num_trials, 3)) for _ in range(6)]
        (
            error_disp_train,
            error_disp_test_1,
            error_disp_test_2,
            error_velo_train,
            error_velo_test_1,
            error_velo_test_2,
        ) = errors_trial

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
                train_set_disp_1, test_set_disp_1, test_set_disp_2, 4, 50000, 6e-5
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
        error_disp_train_mean[i, :] = np.mean(error_disp_train, axis=0)
        error_disp_test_1_mean[i, :] = np.mean(error_disp_test_1, axis=0)
        error_disp_test_2_mean[i, :] = np.mean(error_disp_test_2, axis=0)
        error_velo_train_mean[i, :] = np.mean(error_velo_train, axis=0)
        error_velo_test_1_mean[i, :] = np.mean(error_velo_test_1, axis=0)
        error_velo_test_2_mean[i, :] = np.mean(error_velo_test_2, axis=0)
        error_disp_train_std[i, :] = np.std(error_disp_train, axis=0)
        error_disp_test_1_std[i, :] = np.std(error_disp_test_1, axis=0)
        error_disp_test_2_std[i, :] = np.std(error_disp_test_2, axis=0)
        error_velo_train_std[i, :] = np.std(error_velo_train, axis=0)
        error_velo_test_1_std[i, :] = np.std(error_velo_test_1, axis=0)
        error_velo_test_2_std[i, :] = np.std(error_velo_test_2, axis=0)
        error_disp_train_max[i, :] = np.max(error_disp_train, axis=0)
        error_disp_test_1_max[i, :] = np.max(error_disp_test_1, axis=0)
        error_disp_test_2_max[i, :] = np.max(error_disp_test_2, axis=0)
        error_velo_train_max[i, :] = np.max(error_velo_train, axis=0)
        error_velo_test_1_max[i, :] = np.max(error_velo_test_1, axis=0)
        error_velo_test_2_max[i, :] = np.max(error_velo_test_2, axis=0)
        error_disp_train_min[i, :] = np.min(error_disp_train, axis=0)
        error_disp_test_1_min[i, :] = np.min(error_disp_test_1, axis=0)
        error_disp_test_2_min[i, :] = np.min(error_disp_test_2, axis=0)
        error_velo_train_min[i, :] = np.min(error_velo_train, axis=0)
        error_velo_test_1_min[i, :] = np.min(error_velo_test_1, axis=0)
        error_velo_test_2_min[i, :] = np.min(error_velo_test_2, axis=0)

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
                "error_disp_train_max": error_disp_train_max,
                "error_disp_test_1_max": error_disp_test_1_max,
                "error_disp_test_2_max": error_disp_test_2_max,
                "error_velo_train_max": error_velo_train_max,
                "error_velo_test_1_max": error_velo_test_1_max,
                "error_velo_test_2_max": error_velo_test_2_max,
                "error_disp_train_min": error_disp_train_min,
                "error_disp_test_1_min": error_disp_test_1_min,
                "error_disp_test_2_min": error_disp_test_2_min,
                "error_velo_train_min": error_velo_train_min,
                "error_velo_test_1_min": error_velo_test_1_min,
                "error_velo_test_2_min": error_velo_test_2_min,
            },
            f,
        )


def ae_disp_velo():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
        full_data = pickle.load(f)
    disp = torch.tensor(full_data["displacement"], dtype=torch.float32).to(device)
    velo = torch.tensor(full_data["velocity"], dtype=torch.float32).to(device)
    train_set_disp = disp[:2000, :]
    test_set_disp = disp[2000:10000, :]
    train_set_velo = velo[:2000, :]
    test_set_velo = velo[2000:10000, :]
    ae_disp = AutoEncoder([86, 40, 40, 4])
    ae_velo = AutoEncoder([86, 40, 40, 8])
    ae_dsip_save_path = "./dataset/ae_disp.pth"
    ae_disp_loss_save_path = "./dataset/ae_disp_loss.pkl"
    ae_velo_save_path = "./dataset/ae_velo.pth"
    ae_velo_loss_save_path = "./dataset/ae_velo_loss.pkl"
    ae_disp.train_AE(
        train_set_disp,
        test_set_disp,
        50000,
        6e-5,
        ae_dsip_save_path,
        ae_disp_loss_save_path,
        False,
    )
    ae_velo.train_AE(
        train_set_velo,
        test_set_velo,
        50000,
        1e-4,
        ae_velo_save_path,
        ae_velo_loss_save_path,
        False,
    )
    print("AutoEncoder training finished!")


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
        disp_pred = _dkf(
            start,
            steps,
            [13, 17, 32, 50, 67, 77],
            6,
            1,
            [1e-20, R_value, 3e9],
            "displacement",
        )
        vel_pred = _dkf(
            start,
            steps,
            [13, 17, 32, 50, 67, 77],
            6,
            1,
            [1e-20, R_value, 3e9],
            "velocity",
        )
        disp_rmse[i] = np.sqrt(np.mean((disp_pred - disp) ** 2))
        vel_rmse[i] = np.sqrt(np.mean((vel_pred - velo) ** 2))
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
    with open("./dataset/dkf_eval.pkl", "wb") as f:
        pickle.dump(
            {
                "dkf_disp_1": dkf_disp_1,
                "dkf_velo_1": dkf_velo_1,
                "dkf_disp_2": dkf_disp_2,
                "dkf_velo_2": dkf_velo_2,
            },
            f,
        )


def _rnn(
    ae_model,
    train_set,
    test_set,
    input_size,
    hidden_size,
    output_size,
    num_layers,
    cell_type,
    bidirectional,
    epochs,
    learning_rate,
    model_save_path,
    loss_save_path,
    train_msg,
):
    if ae_model is not None:
        ae_model.eval()
        with torch.no_grad():
            Y_compressed_train = ae_model.encoder(train_set["Y"])
            Y_compressed_test_1 = ae_model.encoder(test_set["Y"])
    else:
        Y_compressed_train = train_set["Y"]
        Y_compressed_test_1 = test_set["Y"]
    _train_set = {"X": train_set["X"], "Y": Y_compressed_train}
    _test_set = {"X": test_set["X"], "Y": Y_compressed_test_1}

    if cell_type == "LSTM":
        _h_0 = torch.zeros(num_layers * (1 + int(bidirectional)), 1, hidden_size)
        _c_0 = torch.zeros(num_layers * (1 + int(bidirectional)), 1, hidden_size)
        _train_hc_0 = (_h_0, _c_0)
        _test_hc_0 = (_h_0, _c_0)

        lstm = Lstm(input_size, hidden_size, output_size, num_layers, bidirectional)
        lstm.train_LSTM(
            train_set=_train_set,
            test_set=_test_set,
            train_hc0=_train_hc_0,
            test_hc0=_test_hc_0,
            epochs=epochs,
            learning_rate=learning_rate,
            model_save_path=model_save_path,
            loss_save_path=loss_save_path,
            train_msg=train_msg,
        )
        return lstm

    elif cell_type == "RNN":
        _h_0 = torch.zeros(num_layers * (1 + int(bidirectional)), 1, hidden_size)
        _train_h_0 = _h_0
        _test_h_0 = _h_0

        rnn = Rnn(input_size, hidden_size, output_size, num_layers, bidirectional)
        rnn.train_RNN(
            train_set=_train_set,
            test_set=_test_set,
            train_h0=_train_h_0,
            test_h0=_test_h_0,
            epochs=epochs,
            learning_rate=learning_rate,
            model_save_path=model_save_path,
            loss_save_path=loss_save_path,
            train_msg=train_msg,
        )
        return rnn


def _mlp(
    ae_model,
    train_set,
    test_set,
    layer_sizes,
    epochs,
    learning_rate,
    model_save_path,
    loss_save_path,
    train_msg,
):
    if ae_model is not None:
        ae_model.eval()
        with torch.no_grad():
            Y_compressed_train = ae_model.encoder(train_set["Y"])
            Y_compressed_test_1 = ae_model.encoder(test_set["Y"])
    else:
        Y_compressed_train = train_set["Y"]
        Y_compressed_test_1 = test_set["Y"]
    _train_set = {"X": train_set["X"], "Y": Y_compressed_train}
    _test_set = {"X": test_set["X"], "Y": Y_compressed_test_1}
    mlp = Mlp(layer_sizes)
    mlp.train_MLP(
        train_set=_train_set,
        test_set=_test_set,
        epochs=epochs,
        learning_rate=learning_rate,
        model_save_path=model_save_path,
        loss_save_path=loss_save_path,
        train_msg=train_msg,
    )


def _ae_models():
    ae_disp = AutoEncoder([86, 40, 40, 4])
    ae_velo = AutoEncoder([86, 40, 40, 8])
    ae_disp.load_state_dict(torch.load("./dataset/ae_disp.pth"))
    ae_velo.load_state_dict(torch.load("./dataset/ae_velo.pth"))
    ae_disp.eval()
    ae_velo.eval()
    return ae_disp, ae_velo


def _data_for_rnn_training(num_ele_per_seg, sensor_idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
        full_data = pickle.load(f)
    disp = torch.tensor(full_data["displacement"], dtype=torch.float32).to(device)
    velo = torch.tensor(full_data["velocity"], dtype=torch.float32).to(device)
    acc = torch.tensor(full_data["acceleration"], dtype=torch.float32).to(device)

    train_set_disp = disp[:num_ele_per_seg, :]
    test_set_disp = disp[num_ele_per_seg:10000, :]
    train_set_velo = velo[:num_ele_per_seg, :]
    test_set_velo = velo[num_ele_per_seg:10000, :]
    train_set_acc = acc[:num_ele_per_seg, sensor_idx]
    test_set_acc = acc[num_ele_per_seg:10000, sensor_idx]

    train_set_acc_2_disp = {"X": train_set_acc, "Y": train_set_disp}
    test_set_acc_2_disp = {"X": test_set_acc, "Y": test_set_disp}
    train_set_acc_2_velo = {"X": train_set_acc, "Y": train_set_velo}
    test_set_acc_2_velo = {"X": test_set_acc, "Y": test_set_velo}

    return (
        train_set_acc_2_disp,
        test_set_acc_2_disp,
        train_set_acc_2_velo,
        test_set_acc_2_velo,
    )


def _data_for_rnn_testing(num_ele_per_seg, sensor_idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./dataset/full_response_excitation_pattern_1.pkl", "rb") as f:
        full_data_1 = pickle.load(f)
    with open("./dataset/full_response_excitation_pattern_2.pkl", "rb") as f:
        full_data_2 = pickle.load(f)
    disp_1 = torch.tensor(full_data_1["displacement"], dtype=torch.float32).to(device)
    velo_1 = torch.tensor(full_data_1["velocity"], dtype=torch.float32).to(device)
    acc_1 = torch.tensor(full_data_1["acceleration"], dtype=torch.float32).to(device)
    disp_2 = torch.tensor(full_data_2["displacement"], dtype=torch.float32).to(device)
    velo_2 = torch.tensor(full_data_2["velocity"], dtype=torch.float32).to(device)
    acc_2 = torch.tensor(full_data_2["acceleration"], dtype=torch.float32).to(device)

    test_set_disp_1 = disp_1[num_ele_per_seg:10000, :]
    test_set_velo_1 = velo_1[num_ele_per_seg:10000, :]
    test_set_acc_1 = acc_1[num_ele_per_seg:10000, sensor_idx]
    test_set_disp_2 = disp_2
    test_set_velo_2 = velo_2
    test_set_acc_2 = acc_2[:, sensor_idx]

    test_set_acc_2_disp_1 = {"X": test_set_acc_1, "Y": test_set_disp_1}
    test_set_acc_2_velo_1 = {"X": test_set_acc_1, "Y": test_set_velo_1}
    test_set_acc_2_disp_2 = {"X": test_set_acc_2, "Y": test_set_disp_2}
    test_set_acc_2_velo_2 = {"X": test_set_acc_2, "Y": test_set_velo_2}

    return (
        test_set_acc_2_disp_1,
        test_set_acc_2_velo_1,
        test_set_acc_2_disp_2,
        test_set_acc_2_velo_2,
    )


def _test_rnn(
    ae_model,
    rnn_model,
    test_set,
    num_ele_per_seg,
    num_seg,
    cell_type,
    h_0,
    c_0,
    pred_save_path,
):
    h_n = h_0
    if cell_type == "LSTM":
        c_n = c_0
    with torch.no_grad():
        for i in range(num_seg):
            input_X = test_set["X"][i * num_ele_per_seg : (i + 1) * num_ele_per_seg, :]
            if cell_type == "LSTM":
                y, (h_n, c_n) = rnn_model.forward(input_X, h_n, c_n)
            elif cell_type == "RNN":
                y, h_n = rnn_model(input_X, h_n)
            if i == 0:
                pred = y
            else:
                pred = torch.cat((pred, y), dim=0)
    pred = ae_model.decoder(pred)
    if pred_save_path is not None:
        pred = pred.cpu().numpy()
        with open(pred_save_path, "wb") as f:
            pickle.dump(pred, f)
    return pred


def birnn_ae_eval(
    num_ele_per_seg=2000,
    acc_idx=[13, 17, 32, 50, 67, 77],
    hidden_size_disp=8,
    hidden_size_velo=16,
):
    ae_disp, ae_velo = _ae_models()
    (
        train_set_acc_2_disp,
        test_set_acc_2_disp,
        train_set_acc_2_velo,
        test_set_acc_2_velo,
    ) = _data_for_rnn_training(num_ele_per_seg, acc_idx)
    model_save_path_disp = "./dataset/birnn_ae_disp.pth"
    loss_save_path_disp = "./dataset/birnn_ae_disp_loss.pkl"
    birnn_disp = _rnn(
        ae_model=ae_disp,
        train_set=train_set_acc_2_disp,
        test_set=test_set_acc_2_disp,
        input_size=len(acc_idx),
        hidden_size=hidden_size_disp,
        output_size=4,
        num_layers=1,
        cell_type="RNN",
        bidirectional=True,
        epochs=500000,
        learning_rate=1e-4,
        model_save_path=model_save_path_disp,
        loss_save_path=loss_save_path_disp,
        train_msg=True,
    )
    model_save_path_velo = "./dataset/birnn_ae_velo.pth"
    loss_save_path_velo = "./dataset/birnn_ae_velo_loss.pkl"
    birnn_velo = _rnn(
        ae_model=ae_velo,
        train_set=train_set_acc_2_velo,
        test_set=test_set_acc_2_velo,
        input_size=len(acc_idx),
        hidden_size=16,
        output_size=8,
        num_layers=1,
        cell_type="RNN",
        bidirectional=True,
        epochs=500000,
        learning_rate=1e-4,
        model_save_path=model_save_path_velo,
        loss_save_path=loss_save_path_velo,
        train_msg=True,
    )
    print("BiRNN training finished!")

    (
        test_set_acc_2_disp_1,
        test_set_acc_2_velo_1,
        test_set_acc_2_disp_2,
        test_set_acc_2_velo_2,
    ) = _data_for_rnn_testing(num_ele_per_seg, acc_idx)

    pred_save_path_1 = "./dataset/birnn_ae_disp_pred_1.pkl"
    pred_save_path_2 = "./dataset/birnn_ae_disp_pred_2.pkl"
    pred_save_path_3 = "./dataset/birnn_ae_velo_pred_1.pkl"
    pred_save_path_4 = "./dataset/birnn_ae_velo_pred_2.pkl"

    _h_0 = torch.zeros(1 * (1 + int(1)), 1, hidden_size_disp)
    pred_disp_1 = _test_rnn(
        ae_disp,
        birnn_disp,
        test_set_acc_2_disp_1,
        num_ele_per_seg,
        4,
        "RNN",
        _h_0,
        None,
        pred_save_path_1,
    )
    pred_disp_2 = _test_rnn(
        ae_disp,
        birnn_disp,
        test_set_acc_2_disp_2,
        num_ele_per_seg,
        5,
        "RNN",
        _h_0,
        None,
        pred_save_path_2,
    )
    _h_0 = torch.zeros(1 * (1 + int(1)), 1, hidden_size_velo)
    pred_velo_1 = _test_rnn(
        ae_velo,
        birnn_velo,
        test_set_acc_2_velo_1,
        num_ele_per_seg,
        4,
        "RNN",
        _h_0,
        None,
        pred_save_path_3,
    )
    pred_velo_2 = _test_rnn(
        ae_velo,
        birnn_velo,
        test_set_acc_2_velo_2,
        num_ele_per_seg,
        5,
        "RNN",
        _h_0,
        None,
        pred_save_path_4,
    )


def rnn_ae_eval(
    num_ele_per_seg=2000,
    acc_idx=[13, 17, 32, 50, 67, 77],
    hidden_size_disp=8,
    hidden_size_velo=16,
):
    ae_disp, ae_velo = _ae_models()
    (
        train_set_acc_2_disp,
        test_set_acc_2_disp,
        train_set_acc_2_velo,
        test_set_acc_2_velo,
    ) = _data_for_rnn_training(num_ele_per_seg, acc_idx)
    model_save_path_disp = "./dataset/rnn_ae_disp.pth"
    loss_save_path_disp = "./dataset/rnn_ae_disp_loss.pkl"
    rnn_disp = _rnn(
        ae_model=ae_disp,
        train_set=train_set_acc_2_disp,
        test_set=test_set_acc_2_disp,
        input_size=len(acc_idx),
        hidden_size=hidden_size_disp,
        output_size=4,
        num_layers=1,
        cell_type="RNN",
        bidirectional=False,
        epochs=500000,
        learning_rate=1e-4,
        model_save_path=model_save_path_disp,
        loss_save_path=loss_save_path_disp,
        train_msg=True,
    )
    model_save_path_velo = "./dataset/rnn_ae_velo.pth"
    loss_save_path_velo = "./dataset/rnn_ae_velo_loss.pkl"
    rnn_velo = _rnn(
        ae_model=ae_velo,
        train_set=train_set_acc_2_velo,
        test_set=test_set_acc_2_velo,
        input_size=len(acc_idx),
        hidden_size=hidden_size_velo,
        output_size=8,
        num_layers=1,
        cell_type="RNN",
        bidirectional=False,
        epochs=500000,
        learning_rate=1e-4,
        model_save_path=model_save_path_velo,
        loss_save_path=loss_save_path_velo,
        train_msg=True,
    )
    print("RNN training finished!")

    (
        test_set_acc_2_disp_1,
        test_set_acc_2_velo_1,
        test_set_acc_2_disp_2,
        test_set_acc_2_velo_2,
    ) = _data_for_rnn_testing(num_ele_per_seg, acc_idx)

    pred_save_path_1 = "./dataset/rnn_ae_disp_pred_1.pkl"
    pred_save_path_2 = "./dataset/rnn_ae_disp_pred_2.pkl"
    pred_save_path_3 = "./dataset/rnn_ae_velo_pred_1.pkl"
    pred_save_path_4 = "./dataset/rnn_ae_velo_pred_2.pkl"

    _h_0 = torch.zeros(1 * (1 + int(0)), 1, hidden_size_disp)
    pred_disp_1 = _test_rnn(
        ae_disp,
        rnn_disp,
        test_set_acc_2_disp_1,
        num_ele_per_seg,
        4,
        "RNN",
        _h_0,
        None,
        pred_save_path_1,
    )
    pred_disp_2 = _test_rnn(
        ae_disp,
        rnn_disp,
        test_set_acc_2_disp_2,
        num_ele_per_seg,
        5,
        "RNN",
        _h_0,
        None,
        pred_save_path_2,
    )
    _h_0 = torch.zeros(1 * (1 + int(0)), 1, hidden_size_velo)
    pred_velo_1 = _test_rnn(
        ae_velo,
        rnn_velo,
        test_set_acc_2_velo_1,
        num_ele_per_seg,
        4,
        "RNN",
        _h_0,
        None,
        pred_save_path_3,
    )
    pred_velo_2 = _test_rnn(
        ae_velo,
        rnn_velo,
        test_set_acc_2_velo_2,
        num_ele_per_seg,
        5,
        "RNN",
        _h_0,
        None,
        pred_save_path_4,
    )


def bilstm_ae_eval(
    num_ele_per_seg=2000,
    acc_idx=[13, 17, 32, 50, 67, 77],
    hidden_size_disp=8,
    hidden_size_velo=16,
):
    ae_disp, ae_velo = _ae_models()
    (
        train_set_acc_2_disp,
        test_set_acc_2_disp,
        train_set_acc_2_velo,
        test_set_acc_2_velo,
    ) = _data_for_rnn_training(num_ele_per_seg, acc_idx)
    model_save_path_disp = "./dataset/bilstm_ae_disp.pth"
    loss_save_path_disp = "./dataset/bilstm_ae_disp_loss.pkl"
    bilstm_disp = _rnn(
        ae_model=ae_disp,
        train_set=train_set_acc_2_disp,
        test_set=test_set_acc_2_disp,
        input_size=len(acc_idx),
        hidden_size=hidden_size_disp,
        output_size=4,
        num_layers=1,
        cell_type="LSTM",
        bidirectional=True,
        epochs=500000,
        learning_rate=1e-4,
        model_save_path=model_save_path_disp,
        loss_save_path=loss_save_path_disp,
        train_msg=True,
    )
    model_save_path_velo = "./dataset/bilstm_ae_velo.pth"
    loss_save_path_velo = "./dataset/bilstm_ae_velo_loss.pkl"
    bilstm_velo = _rnn(
        ae_model=ae_velo,
        train_set=train_set_acc_2_velo,
        test_set=test_set_acc_2_velo,
        input_size=len(acc_idx),
        hidden_size=hidden_size_velo,
        output_size=8,
        num_layers=1,
        cell_type="LSTM",
        bidirectional=True,
        epochs=500000,
        learning_rate=1e-4,
        model_save_path=model_save_path_velo,
        loss_save_path=loss_save_path_velo,
        train_msg=True,
    )
    print("BiLSTM training finished!")

    (
        test_set_acc_2_disp_1,
        test_set_acc_2_velo_1,
        test_set_acc_2_disp_2,
        test_set_acc_2_velo_2,
    ) = _data_for_rnn_testing(num_ele_per_seg, acc_idx)

    pred_save_path_1 = "./dataset/bilstm_ae_disp_pred_1.pkl"
    pred_save_path_2 = "./dataset/bilstm_ae_disp_pred_2.pkl"
    pred_save_path_3 = "./dataset/bilstm_ae_velo_pred_1.pkl"
    pred_save_path_4 = "./dataset/bilstm_ae_velo_pred_2.pkl"

    _h_0 = torch.zeros(1 * (1 + int(1)), 1, hidden_size_disp)
    _c_0 = torch.zeros(1 * (1 + int(1)), 1, hidden_size_disp)
    pred_disp_1 = _test_rnn(
        ae_disp,
        bilstm_disp,
        test_set_acc_2_disp_1,
        num_ele_per_seg,
        4,
        "LSTM",
        _h_0,
        _c_0,
        pred_save_path_1,
    )
    pred_disp_2 = _test_rnn(
        ae_disp,
        bilstm_disp,
        test_set_acc_2_disp_2,
        num_ele_per_seg,
        5,
        "LSTM",
        _h_0,
        _c_0,
        pred_save_path_2,
    )
    _h_0 = torch.zeros(1 * (1 + int(1)), 1, hidden_size_velo)
    _c_0 = torch.zeros(1 * (1 + int(1)), 1, hidden_size_velo)
    pred_velo_1 = _test_rnn(
        ae_velo,
        bilstm_velo,
        test_set_acc_2_velo_1,
        num_ele_per_seg,
        4,
        "LSTM",
        _h_0,
        _c_0,
        pred_save_path_3,
    )
    pred_velo_2 = _test_rnn(
        ae_velo,
        bilstm_velo,
        test_set_acc_2_velo_2,
        num_ele_per_seg,
        5,
        "LSTM",
        _h_0,
        _c_0,
        pred_save_path_4,
    )


def lstm_ae_eval(
    num_ele_per_seg=2000,
    acc_idx=[13, 17, 32, 50, 67, 77],
    hidden_size_disp=8,
    hidden_size_velo=16,
):
    ae_disp, ae_velo = _ae_models()
    (
        train_set_acc_2_disp,
        test_set_acc_2_disp,
        train_set_acc_2_velo,
        test_set_acc_2_velo,
    ) = _data_for_rnn_training(num_ele_per_seg, acc_idx)
    model_save_path_disp = "./dataset/lstm_ae_disp.pth"
    loss_save_path_disp = "./dataset/lstm_ae_disp_loss.pkl"
    lstm_disp = _rnn(
        ae_model=ae_disp,
        train_set=train_set_acc_2_disp,
        test_set=test_set_acc_2_disp,
        input_size=len(acc_idx),
        hidden_size=hidden_size_disp,
        output_size=4,
        num_layers=1,
        cell_type="LSTM",
        bidirectional=False,
        epochs=500000,
        learning_rate=1e-4,
        model_save_path=model_save_path_disp,
        loss_save_path=loss_save_path_disp,
        train_msg=True,
    )
    model_save_path_velo = "./dataset/lstm_ae_velo.pth"
    loss_save_path_velo = "./dataset/lstm_ae_velo_loss.pkl"
    lstm_velo = _rnn(
        ae_model=ae_velo,
        train_set=train_set_acc_2_velo,
        test_set=test_set_acc_2_velo,
        input_size=len(acc_idx),
        hidden_size=hidden_size_velo,
        output_size=8,
        num_layers=1,
        cell_type="LSTM",
        bidirectional=False,
        epochs=500000,
        learning_rate=1e-4,
        model_save_path=model_save_path_velo,
        loss_save_path=loss_save_path_velo,
        train_msg=True,
    )
    print("LSTM training finished!")

    (
        test_set_acc_2_disp_1,
        test_set_acc_2_velo_1,
        test_set_acc_2_disp_2,
        test_set_acc_2_velo_2,
    ) = _data_for_rnn_testing(num_ele_per_seg, acc_idx)

    pred_save_path_1 = "./dataset/lstm_ae_disp_pred_1.pkl"
    pred_save_path_2 = "./dataset/lstm_ae_disp_pred_2.pkl"
    pred_save_path_3 = "./dataset/lstm_ae_velo_pred_1.pkl"
    pred_save_path_4 = "./dataset/lstm_ae_velo_pred_2.pkl"

    _h_0 = torch.zeros(1 * (1 + int(0)), 1, hidden_size_disp)
    _c_0 = torch.zeros(1 * (1 + int(0)), 1, hidden_size_disp)
    pred_disp_1 = _test_rnn(
        ae_disp,
        lstm_disp,
        test_set_acc_2_disp_1,
        num_ele_per_seg,
        4,
        "LSTM",
        _h_0,
        _c_0,
        pred_save_path_1,
    )
    pred_disp_2 = _test_rnn(
        ae_disp,
        lstm_disp,
        test_set_acc_2_disp_2,
        num_ele_per_seg,
        5,
        "LSTM",
        _h_0,
        _c_0,
        pred_save_path_2,
    )
    _h_0 = torch.zeros(1 * (1 + int(0)), 1, hidden_size_velo)
    _c_0 = torch.zeros(1 * (1 + int(0)), 1, hidden_size_velo)
    pred_velo_1 = _test_rnn(
        ae_velo,
        lstm_velo,
        test_set_acc_2_velo_1,
        num_ele_per_seg,
        4,
        "LSTM",
        _h_0,
        _c_0,
        pred_save_path_3,
    )
    pred_velo_2 = _test_rnn(
        ae_velo,
        lstm_velo,
        test_set_acc_2_velo_2,
        num_ele_per_seg,
        5,
        "LSTM",
        _h_0,
        _c_0,
        pred_save_path_4,
    )


def nn_ae_eval(
    num_ele_per_seg=2000,
    acc_idx=[13, 17, 32, 50, 67, 77],
    hidden_size_disp=8,
    hidden_size_velo=16,
):
    "still not sure if this is necessary..."
    ae_disp, ae_velo = _ae_models()
    (
        train_set_acc_2_disp,
        test_set_acc_2_disp,
        train_set_acc_2_velo,
        test_set_acc_2_velo,
    ) = _data_for_rnn_training(num_ele_per_seg, acc_idx)
    model_save_path_disp = "./dataset/nn_ae_disp.pth"
    loss_save_path_disp = "./dataset/nn_ae_disp_loss.pkl"
    raise NotImplementedError


def models_performance_eval():
    dkf_eval()
    birnn_ae_eval()
    rnn_ae_eval()
    bilstm_ae_eval()
    lstm_ae_eval()


def rnn_ae_noise_robustness():
    pass


def rnn_ae_num_sensors():
    pass
