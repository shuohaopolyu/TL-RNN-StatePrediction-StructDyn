from systems import ShearTypeStructure
import numpy as np
import pickle
from utils import compute_metrics, fdd, mac
from models import Rnn, Lstm
import torch
import matplotlib.pyplot as plt
from excitations import FlatNoisePSD, PSDExcitationGenerator
from numpy import linalg as LA


def modal_analysis():
    from pyoma2.algorithm import FSDD_algo
    from pyoma2.OMA import SingleSetup

    data_path = "./dataset/bists/ambient_response.pkl"
    save_path = "./dataset/sts/"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    acc_mtx = solution["acc"].T
    Pali_ss = SingleSetup(acc_mtx, fs=20)
    fsdd = FSDD_algo(name="FSDD", nxseg=1200, method_SD="per", pov=0.5)
    Pali_ss.add_algorithms(fsdd)
    Pali_ss.run_by_name("FSDD")
    Pali_ss.MPE("FSDD", sel_freq=[0.56, 1.49, 2.41, 3.28, 4.02, 4.78], MAClim=0.95)
    ms_array = np.real(Pali_ss["FSDD"].result.Phi)
    nf_array = Pali_ss["FSDD"].result.Fn
    dp_array = Pali_ss["FSDD"].result.Xi
    if save_path is not None:
        file_name = save_path + "modal_analysis.pkl"
        with open(file_name, "wb") as f:
            pickle.dump({"ms": ms_array, "nf": nf_array, "dp": dp_array}, f)
        print("File " + file_name + " saved.")


def model_modal_properties(params):
    stiff_factor = 1e2
    mass_vec = 1 * np.ones(13)
    stiff_vec = (
        np.array([13, 12, 12, 12, 8, 8, 8, 8, 8, 5, 5, 5, 5]) * stiff_factor * params
    )
    # damping ratio set to be 0 here, it has nothing to do with the natural frequencies and mode shapes
    damp_vec = np.zeros_like(mass_vec)
    time = np.array([0, 1, 2, 3, 4])
    acc_g = np.array([0, 0, 0, 0, 0])
    parametric_sts = ShearTypeStructure(
        mass_vec=mass_vec,
        stiff_vec=stiff_vec,
        damp_vec=damp_vec,
        t=time,
        acc_g=acc_g,
    )
    model_nf, model_ms = parametric_sts.freqs_modes()
    return model_nf, model_ms


def loss_function(params, number_of_modes):
    data_path = "./dataset/sts/modal_analysis.pkl"
    model_nf, model_ms = model_modal_properties(params)
    model_nf = model_nf[0:number_of_modes]
    model_ms = model_ms[:, 0:number_of_modes]
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    nf = solution["nf"][0:number_of_modes]
    ms = solution["ms"][:, 0:number_of_modes]
    loss = 0
    for i in range(number_of_modes):
        loss += (1 - model_nf[i] / nf[i]) ** 2 + (1 - mac(model_ms[:, i], ms[:, i]))
    return loss


def model_updating(num_modes=5, method="L-BFGS-B"):
    save_path = "./dataset/sts/"
    from scipy.optimize import minimize

    x0 = np.ones(13)
    obj_func = lambda x: loss_function(x, num_modes)
    res = minimize(obj_func, x0, method=method, options={"disp": True})
    data_path = "./dataset/sts/modal_analysis.pkl"
    model_nf, model_ms = model_modal_properties(res.x)
    model_nf = model_nf[0:num_modes]
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    ms = solution["ms"]
    nf = solution["nf"]
    dp = solution["dp"]
    print((model_nf[0:num_modes] - nf[0:num_modes]) / nf[0:num_modes])
    with open(save_path + "model_updating.pkl", "wb") as f:
        pickle.dump(
            {
                "model_nf": model_nf,
                "model_ms": model_ms,
                "nf": nf,
                "ms": ms,
                "dp": dp,
                "params": res.x,
            },
            f,
        )
    return res.x


def compound_envelope(b1, b2, gamma, t_array):
    tmax = t_array[-1]
    # Assuming t_array is a numpy array
    envelope = np.zeros_like(t_array)
    for i, t in enumerate(t_array):
        normalized_time = t / tmax
        if normalized_time < b1:
            envelope[i] = t / (b1 * tmax)
        elif normalized_time > b2:
            envelope[i] = np.exp(-gamma * (normalized_time - b2))
        else:
            envelope[i] = 1
    return envelope


def seismic_response(
    num=1,
    method="Radau",
    save_path="./dataset/sts/",
):
    # compute the seismic vibration response
    psd_func = FlatNoisePSD(a_v=0.3)
    excitation = PSDExcitationGenerator(psd_func, 40, 10)
    data_path = "./dataset/sts/model_updating.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    params = solution["params"]
    nf = solution["nf"]
    dp = solution["dp"]
    stiff_factor = 1e2
    mass_vec = 1 * np.ones(13)
    stiff_vec = (
        np.array([13, 12, 12, 12, 8, 8, 8, 8, 8, 5, 5, 5, 5]) * stiff_factor * params
    )
    damp_vec = np.array([dp[0], dp[1], nf[0], nf[1]])
    for i_th in range(num):
        time, acc_g = excitation.generate()
        b1 = np.random.uniform(0.1, 0.2)
        gamma = np.random.uniform(3, 5)
        b2 = np.random.uniform(0.4, 0.6)
        window_point = compound_envelope(b1, b2, gamma, time)
        acc_g = acc_g * window_point
        parametric_sts = ShearTypeStructure(
            mass_vec=mass_vec,
            stiff_vec=stiff_vec,
            damp_vec=damp_vec,
            damp_type="Rayleigh",
            t=time,
            acc_g=acc_g,
        )
        # _ = parametric_sts.print_damping_ratio(10)
        # _ = parametric_sts.print_natural_frequency(10)
        acc, velo, disp = parametric_sts.run(method)
        solution = {
            "acc_g": acc_g,
            "time": time,
            "disp": disp,
            "velo": velo,
            "acc": acc,
        }
        if save_path is not None:
            file_name = save_path + "solution" + format(i_th, "03") + ".pkl"
            with open(file_name, "wb") as f:
                pickle.dump(solution, f)
            print("File " + file_name + " saved.")
        else:
            return solution


def validation(method="Radau"):
    time = np.linspace(0, 10, 10000)
    acc = np.sin(2 * np.pi * 1 * time)
    mass_vec = 2 * np.ones(3)
    stiff_vec = 10 * np.ones(3)
    damp_vec = 0.1 * np.ones(3)
    mass_vec[0] = 1
    sts = ShearTypeStructure(
        mass_vec=mass_vec,
        stiff_vec=stiff_vec,
        damp_vec=damp_vec,
        t=time,
        acc_g=acc,
    )
    acc, velo, disp = sts.run(method)
    solution = {
        "acc_g": sts.acc_g,
        "time": time,
        "disp": disp,
        "velo": velo,
        "acc": acc,
    }
    _ = sts.print_damping_ratio(3)
    return solution


def training_test_data(
    acc_sensor, data_compression_ratio=1, num_training_files=40, num_files=100
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disp = []
    velo = []
    acc = []
    state = []
    disp_test = []
    velo_test = []
    acc_test = []
    state_test = []
    for i in range(num_training_files):
        filename = "./dataset/sts/solution" + format(i, "03") + ".pkl"
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        disp = solution["disp"][:, ::data_compression_ratio].T
        velo = solution["velo"][:, ::data_compression_ratio].T
        state.append(np.hstack((disp, velo)))
        acc.append(solution["acc"][acc_sensor, ::data_compression_ratio].T)
    for i in range(num_training_files, num_files):
        filename = "./dataset/sts/solution" + format(i, "03") + ".pkl"
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        disp_test = solution["disp"][:, ::data_compression_ratio].T
        velo_test = solution["velo"][:, ::data_compression_ratio].T
        state_test.append(np.hstack((disp_test, velo_test)))
        acc_test.append(solution["acc"][acc_sensor, ::data_compression_ratio].T)
    state = np.array(state)
    state_test = np.array(state_test)
    acc = np.array(acc)
    acc_test = np.array(acc_test)
    state = torch.tensor(state, dtype=torch.float32).to(device)
    state_test = torch.tensor(state_test, dtype=torch.float32).to(device)
    acc = torch.tensor(acc, dtype=torch.float32).to(device)
    acc_test = torch.tensor(acc_test, dtype=torch.float32).to(device)
    return state, acc, state_test, acc_test


def _birnn(
    acc_sensor,
    data_compression_ratio=1,
    num_training_files=40,
    num_files=100,
    epochs=10000,
    lr=1e-4,
    weight_decay=0.0,
):
    """
    :param acc_sensor: (list) list of accelerometer locations
    :param data_compression_ratio: (int) data compression ratio
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, acc, state_test, acc_test = training_test_data(
        acc_sensor, data_compression_ratio, num_training_files, num_files
    )
    train_set = {"X": acc, "Y": state}
    test_set = {"X": acc_test, "Y": state_test}

    RNN_model_disp = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=True,
    )
    train_h0 = torch.zeros(2, num_training_files, RNN_model_disp.hidden_size).to(device)
    test_h0 = torch.zeros(
        2, num_files - num_training_files, RNN_model_disp.hidden_size
    ).to(device)
    model_save_path = "./dataset/sts/birnn.pth"
    loss_save_path = "./dataset/sts/birnn.pkl"
    train_loss_list, test_loss_list = RNN_model_disp.train_RNN(
        train_set,
        test_set,
        train_h0,
        test_h0,
        epochs=epochs,
        learning_rate=lr,
        model_save_path=model_save_path,
        loss_save_path=loss_save_path,
        train_msg=True,
        weight_decay=weight_decay,
    )
    return train_loss_list, test_loss_list


def _rnn(
    acc_sensor,
    data_compression_ratio=1,
    num_training_files=40,
    num_files=100,
    epochs=10000,
    lr=1e-5,
    weight_decay=0.0,
):
    """
    :param acc_sensor: (list) list of accelerometer locations
    :param data_compression_ratio: (int) data compression ratio
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, acc, state_test, acc_test = training_test_data(
        acc_sensor, data_compression_ratio, num_training_files
    )
    train_set = {"X": acc, "Y": state}
    test_set = {"X": acc_test, "Y": state_test}

    RNN_model_disp = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=False,
    )
    train_h0 = torch.zeros(1, num_training_files, RNN_model_disp.hidden_size).to(device)
    test_h0 = torch.zeros(
        1, num_files - num_training_files, RNN_model_disp.hidden_size
    ).to(device)
    model_save_path = "./dataset/sts/rnn.pth"
    loss_save_path = "./dataset/sts/rnn.pkl"
    train_loss_list, test_loss_list = RNN_model_disp.train_RNN(
        train_set,
        test_set,
        train_h0,
        test_h0,
        epochs=epochs,
        learning_rate=lr,
        model_save_path=model_save_path,
        loss_save_path=loss_save_path,
        train_msg=True,
        weight_decay=weight_decay,
    )
    return train_loss_list, test_loss_list


def build_birnn():
    dr = 1
    ntf = 90
    nf = 100
    acc_sensor = [0, 1, 2, 3, 4]
    _, _ = _birnn(
        acc_sensor,
        data_compression_ratio=dr,
        num_training_files=ntf,
        num_files=nf,
        epochs=50000,
        lr=1e-5,
        weight_decay=0.0,
    )
    RNN4ststate = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=True,
    )
    with open("./dataset/sts/birnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))

    RNN4ststate.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_h0 = torch.zeros(2, nf - ntf, 30).to(device)
    _, _, state_test, acc_test = training_test_data(acc_sensor, dr, ntf)
    with torch.no_grad():
        state_pred, _ = RNN4ststate(acc_test, test_h0)
    state_pred = state_pred.cpu().numpy()
    state_pred = state_pred[:, :, 8]
    state_test = state_test.cpu().numpy()
    state_test = state_test[:, :, 8]
    plt.plot(state_test[3, :], label="Ground truth")
    plt.plot(state_pred[3, :], label="Prediction", linestyle="--")
    plt.legend()
    plt.show()


def build_rnn():
    dr = 1
    ntf = 90
    nf = 100
    acc_sensor = [0, 1, 2, 3, 4]
    _, _ = _rnn(
        acc_sensor,
        data_compression_ratio=dr,
        num_training_files=ntf,
        num_files=nf,
        epochs=50000,
        lr=1e-5,
    )
    RNN4ststate = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=False,
    )
    with open("./dataset/sts/rnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    RNN4ststate.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_h0 = torch.zeros(1, nf - ntf, 30).to(device)
    _, _, state_test, acc_test = training_test_data(acc_sensor, dr, ntf)
    with torch.no_grad():
        state_pred, _ = RNN4ststate(acc_test, test_h0)
    state_pred = state_pred.cpu().numpy()
    state_pred = state_pred[:, :, 16]
    state_test = state_test.cpu().numpy()
    state_test = state_test[:, :, 16]
    plt.plot(state_test[4, :], label="Ground truth")
    plt.plot(state_pred[4, :], label="Prediction", linestyle="--")
    plt.legend()
    plt.show()


def _dkf(
    acc_sensor,
    data_compression_ratio,
    num_training_files,
    num_modes,
    dkf_params=[1e-20, 1e-2, 3e9],
    type="test",
):
    if type == "test":
        _, _, state_full, acc_full = training_test_data(
            acc_sensor, data_compression_ratio, num_training_files
        )
    else:
        state_full, acc_full, _, _ = training_test_data(
            acc_sensor, data_compression_ratio, num_training_files
        )
    state_full = state_full.cpu().numpy()
    acc_full = acc_full.cpu().numpy()
    steps = state_full.shape[1]
    # Define the system
    data_path = "./dataset/sts/model_updating.pkl"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    params = solution["params"]
    nf = solution["nf"]
    dp = solution["dp"]
    stiff_factor = 1e2
    mass_vec = 1 * np.ones(13)
    stiff_vec = (
        np.array([13, 12, 12, 12, 8, 8, 8, 8, 8, 5, 5, 5, 5]) * stiff_factor * params
    )
    damp_vec = np.array([dp[0], dp[1], nf[0], nf[1]])
    time = np.arange(0, steps/20, 1/20)
    acc_g = np.zeros_like(time)
    parametric_sts = ShearTypeStructure(
        mass_vec=mass_vec,
        stiff_vec=stiff_vec,
        damp_vec=damp_vec,
        damp_type="Rayleigh",
        t=time,
        acc_g=acc_g,
    )
    parametric_sts.resp_dof = acc_sensor
    parametric_sts.n_s = len(acc_sensor)
    _, modes = parametric_sts.freqs_modes(mode_normalize=True)
    md_mtx = modes[:, 0:num_modes]

    # compute the state space matrices
    A, B, G, J = parametric_sts.truncated_state_space_mtx(
        truncation=num_modes, type="discrete"
    )
    Q_zeta = np.eye(num_modes * 2) * dkf_params[0]
    R = np.eye(len(acc_sensor)) * dkf_params[1]
    Q_p = np.eye(1) * dkf_params[2]

    # initialization
    load_mtx = np.zeros((1, steps))
    zeta_mtx = np.zeros((num_modes * 2, steps))
    P_p_mtx = np.zeros((1, 1, steps))
    P_mtx = np.zeros((num_modes * 2, num_modes * 2, steps))
    P_p_mtx[:, :, 0] = Q_p
    P_mtx[:, :, 0] = Q_zeta
    disp_list = []
    velo_list = []
    if type == "test":
        it_num = 100 - num_training_files
    else:
        it_num = num_training_files

    for j in range(it_num):
        obs_data = acc_full[j, :, :].T
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
            # print progress every 10% for test
            # progress_percentage = (i + 2) / (steps) * 100
            # if progress_percentage % 10 == 0:
            #     print(f"Progress: {progress_percentage:.0f}%")
        disp_pred = md_mtx @ zeta_mtx[:num_modes, :]
        disp_list.append(disp_pred.T)
        print(
            "Dual Kalman Filter finished for {order}-th displacement prediction!".format(
                order=j + 1
            )
        )
        # return disp_pred
        velo_pred = md_mtx @ zeta_mtx[num_modes:, :]
        velo_list.append(velo_pred.T)
        print(
            "Dual Kalman Filter finished for {order}-th velocity prediction!".format(
                order=j + 1
            )
        )
        # return vel_pred
    disp_array = np.array(disp_list)
    velo_array = np.array(velo_list)
    return disp_array, velo_array


def tune_dkf_params():
    R_values = np.logspace(-6, 2, 10)
    disp_rmse = np.zeros(len(R_values))
    velo_rmse = np.zeros(len(R_values))
    state_full, _, _, _ = training_test_data([0, 1, 2, 3, 4], 1, 1)
    disp = state_full[:, :, 0:13].cpu().numpy()
    velo = state_full[:, :, 13:26].cpu().numpy()
    for i, R_value in enumerate(R_values):
        disp_pred, velo_pred = _dkf(
            [0, 1, 2, 3, 4], 1, 1, 13, [1e-20, R_value, 1], "training"
        )
        disp_rmse[i] = np.sqrt(np.mean((disp_pred - disp) ** 2))
        velo_rmse[i] = np.sqrt(np.mean((velo_pred - velo) ** 2))
    plt.figure()
    plt.plot(disp_pred[0, :, 8], label="Prediction")
    plt.plot(disp[0, :, 8], label="Ground truth")
    plt.show()
    plt.figure()
    plt.plot(velo_pred[0, :, 8], label="Prediction")
    plt.plot(velo[0, :, 8], label="Ground truth")
    plt.show()
    results = {"R_values": R_values, "disp_rmse": disp_rmse, "vel_rmse": velo_rmse}
    with open("./dataset/sts/dkf_tune.pkl", "wb") as f:
        pickle.dump(results, f)


def dkf():
    # parameters are tuned via the funtion tune_dkf_params
    disp_pred, velo_pred = _dkf(
        [0, 1, 2, 3, 4], 1, 90, 13, [1e-20, 1e-3, 1], "test"
    )
    with open("./dataset/sts/dkf_pred.pkl", "wb") as f:
        pickle.dump({"disp_pred": disp_pred, "velo_pred": velo_pred}, f)

def generate_seismic_response(acc_sensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_seismic = 1
    acc_array = []
    state_array = []
    for i in range(num_seismic):
        file_name = (
            "./dataset/bists/solution"
            + format(i + 4, "03")
            + ".pkl"
        )
        with open(file_name, "rb") as f:
            solution = pickle.load(f)
        acc = solution["acc"][acc_sensor, :].T
        disp = solution["disp"].T
        velo = solution["velo"].T
        acc[:, 1:] = acc[:, 1:] + np.tile(acc[:, 0].reshape(-1,1), (1, acc.shape[1] - 1))
        disp[:, 1:] = disp[:, 1:] + np.tile(disp[:, 0].reshape(-1,1), (1, disp.shape[1] - 1))
        velo[:, 1:] = velo[:, 1:] + np.tile(velo[:, 0].reshape(-1,1), (1, velo.shape[1] - 1))
        state = np.hstack((disp, velo))
        time = solution["time"]
        time_interpolated = np.arange(time[0], time[-1], 1/20)
        acc_interpolated = np.zeros((len(time_interpolated), acc.shape[1]))
        state_interpolated = np.zeros((len(time_interpolated), state.shape[1]))
        for j in range(acc.shape[1]):
            acc_interpolated[:, j] = np.interp(time_interpolated, time, acc[:, j])
        for j in range(state.shape[1]):
            state_interpolated[:, j] = np.interp(time_interpolated, time, state[:, j])
        acc_array.append(acc_interpolated)
        state_array.append(state_interpolated)
    acc_array = np.array(acc_array)
    state_array = np.array(state_array)
    acc_tensor = torch.tensor(acc_array, dtype=torch.float32).to(device)
    state_tensor = torch.tensor(state_array, dtype=torch.float32).to(device)
    return acc_tensor, state_tensor


def birnn_seismic_pred():
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 1
    RNN4ststate = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=True,
    )
    with open("./dataset/sts/birnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))

    RNN4ststate.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_h0 = torch.zeros(2, num_seismic, 30).to(device)
    acc_tensor, state_tensor = generate_seismic_response(acc_sensor)
    with torch.no_grad():
        state_pred, _ = RNN4ststate(acc_tensor, test_h0)
    state_pred = state_pred.cpu().numpy()
    state_pred = state_pred[:, :, 8]
    state_tensor = state_tensor.cpu().numpy()
    state_tensor = state_tensor[:, :, 8]
    plt.plot(state_tensor[0, :], label="Ground truth")
    plt.plot(state_pred[0, :], label="Prediction", linestyle="--")
    plt.legend()
    plt.show()

def rnn_seismic_pred():
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 1
    RNN4ststate = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=26,
        num_layers=1,
        bidirectional=False,
    )
    with open("./dataset/sts/rnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))

    RNN4ststate.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_h0 = torch.zeros(1, num_seismic, 30).to(device)
    acc_tensor, state_tensor = generate_seismic_response(acc_sensor)
    with torch.no_grad():
        state_pred, _ = RNN4ststate(acc_tensor, test_h0)
    state_pred = state_pred.cpu().numpy()
    state_pred = state_pred[:, :, 16]
    state_tensor = state_tensor.cpu().numpy()
    state_tensor = state_tensor[:, :, 16]
    plt.plot(state_tensor[0, :], label="Ground truth")
    plt.plot(state_pred[0, :], label="Prediction", linestyle="--")
    plt.legend()
    plt.show()






