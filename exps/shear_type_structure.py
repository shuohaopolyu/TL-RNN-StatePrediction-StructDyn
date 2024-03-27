from systems import ShearTypeStructure
import numpy as np
import pickle
from utils import mac
from models import Rnn
import torch
import matplotlib.pyplot as plt
from excitations import FlatNoisePSD, PSDExcitationGenerator
from numpy import linalg as LA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


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
    print(1/nf_array)
    print(dp_array)
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
    acc_sensor,
    data_compression_ratio=1,
    num_training_files=40,
    num_files=100,
    output="all",
):
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
        if output == "all":
            state.append(np.hstack((disp, velo)))
        elif output == "disp":
            state.append(disp)
        elif output == "velo":
            state.append(velo)
        acc.append(solution["acc"][acc_sensor, ::data_compression_ratio].T)
    for i in range(num_training_files, num_files):
        filename = "./dataset/sts/solution" + format(i, "03") + ".pkl"
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        disp_test = solution["disp"][:, ::data_compression_ratio].T
        velo_test = solution["velo"][:, ::data_compression_ratio].T
        if output == "all":
            state_test.append(np.hstack((disp_test, velo_test)))
        elif output == "disp":
            state_test.append(disp_test)
        elif output == "velo":
            state_test.append(velo_test)
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
    output="all",
):
    """
    :param acc_sensor: (list) list of accelerometer locations
    :param data_compression_ratio: (int) data compression ratio
    """
    state, acc, state_test, acc_test = training_test_data(
        acc_sensor, data_compression_ratio, num_training_files, num_files, output
    )
    train_set = {"X": acc, "Y": state}
    test_set = {"X": acc_test, "Y": state_test}
    output_size = 26 if output == "all" else 13
    RNN_model_disp = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
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


def build_birnn(output="all"):
    dr = 1
    ntf = 90
    nf = 100
    acc_sensor = [0, 1, 2, 3, 4]
    output_size = 26 if output == "all" else 13
    epochs = 60000 if output == "all" else 40000
    _, _ = _birnn(
        acc_sensor,
        data_compression_ratio=dr,
        num_training_files=ntf,
        num_files=nf,
        epochs=epochs,
        lr=8e-6,
        weight_decay=0.0,
        output=output,
    )
    RNN4ststate = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=True,
    )
    with open("./dataset/sts/birnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))

    RNN4ststate.eval()
    test_h0 = torch.zeros(2, nf - ntf, 30).to(device)
    _, _, state_test, acc_test = training_test_data(acc_sensor, dr, ntf, nf, output)
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
        epochs=80000,
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


def _dkf(obs_data, system_mtx, dkf_params):
    """Dual Kalman Filter for state estimation in structural dynamics

    Args:
        obs_data (numpy array): observation data, shape (n_sensor, n_t)
        system_mtx (dict): system matrices, keys: A, B, G, J, where A, B, G, J are numpy arrays
        akf_params (dict): dual Kalman filter parameters, keys: Q_zeta, R, Q_p, where Q_zeta, R, Q_p are numpy arrays

    Returns:
        x_mtx (numpy array): stateestimation, shape (n_state, n_t)
        p_mtx (numpy array): input estimation, shape (n_force, n_t)
    """
    steps = obs_data.shape[1]
    A, B, C, D = system_mtx["A"], system_mtx["B"], system_mtx["C"], system_mtx["D"]
    num_state = A.shape[1]
    num_force = B.shape[1]
    Q_zeta, R, Q_p = dkf_params["Q_zeta"], dkf_params["R"], dkf_params["Q_p"]
    P_p_mtx = np.zeros((num_force, num_force, steps))
    P_mtx = np.zeros((num_state, num_state, steps))
    P_p_mtx[:, :, 0] = Q_p
    P_mtx[:, :, 0] = Q_zeta
    x_mtx = np.zeros((num_state, steps))
    p_mtx = np.zeros((num_force, steps))
    for i in range(steps - 1):
        p_k_m = p_mtx[:, i]
        P_k_pm = P_p_mtx[:, :, i] + Q_p
        G_k_p = P_k_pm @ D.T @ LA.inv(D @ P_k_pm @ D.T + R)
        p_mtx[:, i + 1] = p_k_m + G_k_p @ (
            obs_data[:, i + 1] - C @ x_mtx[:, i] - D @ p_k_m
        )
        P_p_mtx[:, :, i + 1] = P_k_pm - G_k_p @ D @ P_k_pm
        x_k_m = A @ x_mtx[:, i] + B @ p_mtx[:, i + 1]
        P_k_m = A @ P_mtx[:, :, i] @ A.T + Q_zeta
        G_k_zeta = P_k_m @ C.T @ LA.inv(C @ P_k_m @ C.T + R)
        x_mtx[:, i + 1] = x_k_m + G_k_zeta @ (obs_data[:, i + 1] - C @ x_k_m)
        P_mtx[:, :, i + 1] = P_k_m - G_k_zeta @ C @ P_k_m
    return x_mtx, p_mtx


def _akf(obs_data, system_mtx, akf_params):
    """Augmented Kalman Filter for state estimation in structural dynamics

    Args:
        obs_data (numpy array): observation data, shape (n_sensor, n_t)
        system_mtx (dict): system matrices, keys: A, B, C, D, where A, B, C, D are numpy arrays
        akf_params (dict): augmented Kalman filter parameters, keys: Q_zeta, R, Q_p, where Q_zeta, R, Q_p are numpy arrays

    Returns:
        x_mtx (numpy array): state and input estimation, shape (n_state + num_force, n_t)
    """
    steps = obs_data.shape[1]
    A, B, C, D = system_mtx["A"], system_mtx["B"], system_mtx["C"], system_mtx["D"]
    num_state = A.shape[1]
    num_force = B.shape[1]
    Q_zeta, R, Q_p = akf_params["Q_zeta"], akf_params["R"], akf_params["Q_p"]
    A_a = np.block([[A, B], [np.zeros((num_force, num_state)), np.eye(num_force)]])
    C_a = np.hstack([C, D])
    Q = np.block(
        [[Q_zeta, np.zeros((A.shape[0], 1))], [np.zeros((num_force, A.shape[0])), Q_p]]
    )
    P_mtx = np.zeros((num_state + num_force, num_state + num_force, steps))
    P_mtx[:, :, 0] = Q
    x_mtx = np.zeros((num_state + num_force, steps))
    for i in range(steps - 1):
        Lk = P_mtx[:, :, i] @ C_a.T @ LA.inv(C_a @ P_mtx[:, :, i] @ C_a.T + R)
        x_hat = x_mtx[:, i] + Lk @ (obs_data[:, i + 1] - C_a @ x_mtx[:, i])
        Pkk = P_mtx[:, :, i] - Lk @ C_a @ P_mtx[:, :, i]
        x_mtx[:, i + 1] = A_a @ x_hat
        P_mtx[:, :, i + 1] = A_a @ Pkk @ A_a.T + Q
    return x_mtx


def _system_matrices(acc_sensor, num_modes, kwargs=None):
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
    time = np.arange(0, 1, 1 / 20)
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
    A, B, C, D = parametric_sts.truncated_state_space_mtx(
        truncation=num_modes, type="discrete", kwargs=kwargs
    )
    system_mtx = {"A": A, "B": B, "C": C, "D": D}
    return system_mtx, md_mtx


def build_dkf(
    acc_sensor=[0, 1, 2, 3, 4],
    data_compression_ratio=1,
    num_training_files=90,
    num_modes=13,
    dkf_params=[1e-20, 1e-3, 1],
    type="test",
):
    """Dual Kalman Filter for state estimation using test data

    Args:
        acc_sensor (list): accele. Defaults to [0, 1, 2, 3, 4].
        data_compression_ratio (int): _description_. Defaults to 1.
        num_training_files (int): _description_. Defaults to 90.
        num_modes (int): _description_. Defaults to 13.
        dkf_params (list): _description_. Defaults to [1e-20, 1e-3, 1].
        type (str): _description_. Defaults to "test".
    """
    # parameters are tuned via the funtion tune_dkf_params
    if type == "test":
        _, _, _, acc_full = training_test_data(
            acc_sensor, data_compression_ratio, num_training_files
        )
    else:
        _, acc_full, _, _ = training_test_data(
            acc_sensor, data_compression_ratio, num_training_files
        )
    acc_full = acc_full.cpu().numpy()
    steps = acc_full.shape[1]
    # Define the system
    system_mtx, md_mtx = _system_matrices(acc_sensor, num_modes)
    Q_zeta = np.eye(num_modes * 2) * dkf_params[0]
    R = np.eye(len(acc_sensor)) * dkf_params[1]
    Q_p = np.eye(1) * dkf_params[2]
    dkf_params = {"Q_zeta": Q_zeta, "R": R, "Q_p": Q_p}
    disp_pred = np.zeros((acc_full.shape[0], steps, num_modes))
    velo_pred = np.zeros((acc_full.shape[0], steps, num_modes))
    for i in range(acc_full.shape[0]):
        obs_data = acc_full[i, :, :].T
        x_mtx, _ = _dkf(obs_data, system_mtx, dkf_params)
        disp_mtx = md_mtx @ x_mtx[:num_modes, :]
        velo_mtx = md_mtx @ x_mtx[num_modes : num_modes * 2, :]
        disp_pred[i, :, :] = disp_mtx.T
        velo_pred[i, :, :] = velo_mtx.T
    with open("./dataset/sts/dkf_pred.pkl", "wb") as f:
        pickle.dump({"disp_pred": disp_pred, "velo_pred": velo_pred}, f)


def build_akf(
    acc_sensor=[0, 1, 2, 3, 4],
    data_compression_ratio=1,
    num_training_files=90,
    num_modes=13,
    akf_params=[1e-20, 1e-3, 1],
    type="test",
):
    """Augumented Kalman Filter for state estimation using test data

    Args:
        acc_sensor (list, optional): _description_. Defaults to [0, 1, 2, 3, 4].
        data_compression_ratio (int, optional): _description_. Defaults to 1.
        num_training_files (int, optional): _description_. Defaults to 90.
        num_modes (int, optional): _description_. Defaults to 13.
        akf_params (list, optional): _description_. Defaults to [1e-20, 1e-3, 1].
        type (str, optional): _description_. Defaults to "test".
    """
    if type == "test":
        _, _, _, acc_full = training_test_data(
            acc_sensor, data_compression_ratio, num_training_files
        )
    else:
        _, acc_full, _, _ = training_test_data(
            acc_sensor, data_compression_ratio, num_training_files
        )
    acc_full = acc_full.cpu().numpy()
    steps = acc_full.shape[1]
    # Define the system
    system_mtx, md_mtx = _system_matrices(acc_sensor, num_modes)
    Q_zeta = np.eye(num_modes * 2) * akf_params[0]
    R = np.eye(len(acc_sensor)) * akf_params[1]
    Q_p = np.eye(1) * akf_params[2]
    akf_params = {"Q_zeta": Q_zeta, "R": R, "Q_p": Q_p}
    disp_pred = np.zeros((acc_full.shape[0], steps, num_modes))
    velo_pred = np.zeros((acc_full.shape[0], steps, num_modes))
    for i in range(acc_full.shape[0]):
        obs_data = acc_full[i, :, :].T
        x_mtx = _akf(obs_data, system_mtx, akf_params)
        disp_mtx = md_mtx @ x_mtx[:num_modes, :]
        velo_mtx = md_mtx @ x_mtx[num_modes : num_modes * 2, :]
        disp_pred[i, :, :] = disp_mtx.T
        velo_pred[i, :, :] = velo_mtx.T
    with open("./dataset/sts/akf_pred.pkl", "wb") as f:
        pickle.dump({"disp_pred": disp_pred, "velo_pred": velo_pred}, f)


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


def generate_seismic_response(acc_sensor, num_seismic, output="all"):
    """
    generate seismic response from the high-fidelity model (real-world structure)

    Args:
        acc_sensor (list): a list of accelerometer locations
        num_seismic (int): number of seismic response to be generated

    Returns:
        acc_list (list): a list containing num_seismic acceleration tensors, each tensor has the shape of (n_t, n_sensor)
        state_list (list): a list containing num_seismic state tensors, each tensor has the shape of (n_t, n_state)
    """

    acc_list = []
    state_list = []
    for i in range(num_seismic):
        file_name = "./dataset/bists/solution" + format(i, "03") + ".pkl"
        with open(file_name, "rb") as f:
            solution = pickle.load(f)
        acc = solution["acc"][acc_sensor, :].T
        disp = solution["disp"].T
        velo = solution["velo"].T
        acc[:, 1:] = acc[:, 1:] + np.tile(
            acc[:, 0].reshape(-1, 1), (1, acc.shape[1] - 1)
        )
        disp[:, 1:] = disp[:, 1:] + np.tile(
            disp[:, 0].reshape(-1, 1), (1, disp.shape[1] - 1)
        )
        velo[:, 1:] = velo[:, 1:] + np.tile(
            velo[:, 0].reshape(-1, 1), (1, velo.shape[1] - 1)
        )
        if output == "all":
            state = np.hstack((disp, velo))
        elif output == "disp":
            state = disp
        elif output == "velo":
            state = velo
        time = solution["time"]
        time_interpolated = np.arange(time[0], time[-1], 1 / 20)
        acc_interpolated = np.zeros((len(time_interpolated), acc.shape[1]))
        state_interpolated = np.zeros((len(time_interpolated), state.shape[1]))
        for j in range(acc.shape[1]):
            acc_interpolated[:, j] = np.interp(time_interpolated, time, acc[:, j])
        for j in range(state.shape[1]):
            state_interpolated[:, j] = np.interp(time_interpolated, time, state[:, j])
        acc_interpolated = torch.tensor(acc_interpolated, dtype=torch.float32).to(
            device
        )
        state_interpolated = torch.tensor(state_interpolated, dtype=torch.float32).to(
            device
        )
        acc_list.append(acc_interpolated)
        state_list.append(state_interpolated)
    return acc_list, state_list


def generate_floor_drift(num_seismic, floors):
    # a good paper: A CRITICAL ASSESSMENT OF INTERSTORY DRIFT MEASUREMENTS
    """
    generate floor drifts from the seismic response

    Args:
        num_seismic (int): number of seismic response to be generated
        floors (list): a list of floor pairs, e.g., [[1,2], [3,4], [5,6], [9,11]]

    Returns:
        drift_list (list): a list of tensors, each of shape (nt, len(floors))
    """
    drift_list = []

    for i in range(num_seismic):
        file_name = "./dataset/bists/solution" + format(i, "03") + ".pkl"
        with open(file_name, "rb") as f:
            solution = pickle.load(f)
        disp = solution["disp"].T
        disp[:, 1:] = disp[:, 1:] + np.tile(
            disp[:, 0].reshape(-1, 1), (1, disp.shape[1] - 1)
        )
        time = solution["time"]
        time_interpolated = np.arange(time[0], time[-1], 1 / 20)
        disp_interpolated = np.zeros((len(time_interpolated), disp.shape[1]))
        for j in range(disp.shape[1]):
            disp_interpolated[:, j] = np.interp(time_interpolated, time, disp[:, j])
        drift_mtx = np.zeros((len(time_interpolated), len(floors)))
        for j in range(len(floors)):
            drift_mtx[:, j] = (
                disp_interpolated[:, floors[j][1]] - disp_interpolated[:, floors[j][0]]
            )
        drift_tensor = torch.tensor(drift_mtx, dtype=torch.float32).to(device)
        drift_list.append(drift_tensor)
    return drift_list


def floor_drift_pred(RNN4ststate, acc_tensor, floors):
    """compute floor drifts from the rnn and birnn predictions

    Args:
        RNN4ststate (obj): rnn or birnn model
        acc_tensor (tensor): acceleration tensor of shape (nt, n_sensor)
        floors (list): a list of floor pairs, e.g., [[1,2], [3,4], [5,6], [9,11]]

    Returns:
        drift_pred: a tensor of shape (nt, len(floors))
    """
    if RNN4ststate.bidirectional:
        h0 = torch.zeros(2, 1, 30).to(device)
    else:
        h0 = torch.zeros(1, 1, 30).to(device)
    state_pred, _ = RNN4ststate(acc_tensor, h0)
    state_pred = state_pred.squeeze()
    disp_pred = state_pred[:, 0:13]
    drift_pred = torch.zeros(disp_pred.shape[0], len(floors)).to(device)
    for i in range(len(floors)):
        drift_pred[:, i] = disp_pred[:, floors[i][1]] - disp_pred[:, floors[i][0]]
    return drift_pred


def birnn_seismic_pred(output="all"):
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 4
    output_size = 26 if output == "all" else 13
    RNN4ststate = Rnn(
        input_size=len(acc_sensor),
        hidden_size=30,
        output_size=output_size,
        num_layers=1,
        bidirectional=True,
    )
    with open("./dataset/sts/birnn.pth", "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    RNN4ststate.eval()
    test_h0 = torch.zeros(2, 1, 30).to(device)
    acc_list, state_list = generate_seismic_response(acc_sensor, num_seismic)
    state_pred_list = []
    with torch.no_grad():
        for i in range(num_seismic):
            acc_tensor = acc_list[i].unsqueeze(0)
            state_pred, _ = RNN4ststate(acc_tensor, test_h0)
            state_pred = state_pred.squeeze()
            state_pred = state_pred.cpu().numpy()
            state_pred_list.append(state_pred)
            i_th_state_true = state_list[i].cpu().numpy()
            i_th_state_pred = state_pred_list[i]
            plt.plot(i_th_state_true[:, 6], label="Ground truth")
            plt.plot(i_th_state_pred[:, 6], label="Prediction", linestyle="--")
            plt.legend()
            plt.show()


def rnn_seismic_pred():
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 4
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
    test_h0 = torch.zeros(1, 1, 30).to(device)
    acc_list, state_list = generate_seismic_response(acc_sensor, num_seismic)
    state_pred_list = []
    with torch.no_grad():
        for i in range(num_seismic):
            acc_tensor = acc_list[i].unsqueeze(0)
            state_pred, _ = RNN4ststate(acc_tensor, test_h0)
            state_pred = state_pred.squeeze()
            state_pred = state_pred.cpu().numpy()
            state_pred_list.append(state_pred)
            i_th_state_true = state_list[i].cpu().numpy()
            i_th_state_pred = state_pred_list[i]
            plt.plot(i_th_state_true[:, 8], label="Ground truth")
            plt.plot(i_th_state_pred[:, 8], label="Prediction", linestyle="--")
            plt.legend()
            plt.show()


def dkf_seismic_pred(
    acc_sensor=[0, 1, 2, 3, 4],
    num_modes=13,
    dkf_params=[1e-20, 1e-3, 1],
    num_seismic=4,
):

    acc_list, _ = generate_seismic_response(acc_sensor, num_seismic)
    # Define the system
    system_mtx, md_mtx = _system_matrices(acc_sensor, num_modes)
    Q_zeta = np.eye(num_modes * 2) * dkf_params[0]
    R = np.eye(len(acc_sensor)) * dkf_params[1]
    Q_p = np.eye(1) * dkf_params[2]
    dkf_params = {"Q_zeta": Q_zeta, "R": R, "Q_p": Q_p}
    disp_list = []
    velo_list = []
    for i in range(num_seismic):
        steps = acc_list[i].shape[0]
        disp_pred = np.zeros((steps, num_modes))
        velo_pred = np.zeros((steps, num_modes))
        obs_data = acc_list[i].cpu().numpy().T
        x_mtx, _ = _dkf(obs_data, system_mtx, dkf_params)
        disp_mtx = md_mtx @ x_mtx[:num_modes, :]
        velo_mtx = md_mtx @ x_mtx[num_modes : num_modes * 2, :]
        disp_pred[:, :] = disp_mtx.T
        velo_pred[:, :] = velo_mtx.T
        disp_list.append(disp_pred)
        velo_list.append(velo_pred)
    return disp_list, velo_list


def integr_dkf_seismic_pred(
    acc_sensor=[0, 1, 2, 3, 4],
    num_modes=13,
    dkf_params=[1e-20, 1e-3, 1],
    num_seismic=4,
    floors=[[0, 1], [1, 2], [2, 3], [3, 4]],
):
    acc_list, _ = generate_seismic_response(acc_sensor, num_seismic)
    floor_drift = generate_floor_drift(num_seismic, floors)
    # Define the system
    kwargs = {"floors": floors}
    system_mtx, md_mtx = _system_matrices(acc_sensor, num_modes, kwargs=kwargs)
    Q_zeta = np.eye(num_modes * 2) * dkf_params[0]
    R = np.eye(len(acc_sensor) + len(floors)) * dkf_params[1]
    Q_p = np.eye(1) * dkf_params[2]
    dkf_params = {"Q_zeta": Q_zeta, "R": R, "Q_p": Q_p}
    disp_list = []
    velo_list = []
    for i in range(num_seismic):
        steps = acc_list[i].shape[0]
        disp_pred = np.zeros((steps, num_modes))
        velo_pred = np.zeros((steps, num_modes))
        obs_data = np.vstack(
            (
                floor_drift[i].cpu().numpy().T,
                acc_list[i].cpu().numpy().T,
            )
        )
        x_mtx, _ = _dkf(obs_data, system_mtx, dkf_params)
        disp_mtx = md_mtx @ x_mtx[:num_modes, :]
        velo_mtx = md_mtx @ x_mtx[num_modes : num_modes * 2, :]
        disp_pred[:, :] = disp_mtx.T
        velo_pred[:, :] = velo_mtx.T
        disp_list.append(disp_pred)
        velo_list.append(velo_pred)
    return disp_list, velo_list


def akf_seismic_pred(
    acc_sensor=[0, 1, 2, 3, 4],
    num_modes=13,
    dkf_params=[1e-20, 1e-3, 1],
    num_seismic=4,
):
    acc_list, _ = generate_seismic_response(acc_sensor, num_seismic)
    # Define the system
    system_mtx, md_mtx = _system_matrices(acc_sensor, num_modes)
    Q_zeta = np.eye(num_modes * 2) * dkf_params[0]
    R = np.eye(len(acc_sensor)) * dkf_params[1]
    Q_p = np.eye(1) * dkf_params[2]
    dkf_params = {"Q_zeta": Q_zeta, "R": R, "Q_p": Q_p}
    disp_list = []
    velo_list = []
    for i in range(num_seismic):
        steps = acc_list[i].shape[0]
        disp_pred = np.zeros((steps, num_modes))
        velo_pred = np.zeros((steps, num_modes))
        obs_data = acc_list[i].cpu().numpy().T
        x_mtx = _akf(obs_data, system_mtx, dkf_params)
        disp_mtx = md_mtx @ x_mtx[:num_modes, :]
        velo_mtx = md_mtx @ x_mtx[num_modes : num_modes * 2, :]
        disp_pred[:, :] = disp_mtx.T
        velo_pred[:, :] = velo_mtx.T
        disp_list.append(disp_pred)
        velo_list.append(velo_pred)
    return disp_list, velo_list


def integr_akf_seismic_pred(
    acc_sensor=[0, 1, 2, 3, 4],
    num_modes=13,
    dkf_params=[1e-20, 1e-3, 1],
    num_seismic=4,
    floors=[[0, 1], [1, 2], [2, 3], [3, 4]],
):
    acc_list, _ = generate_seismic_response(acc_sensor, num_seismic)
    floor_drift = generate_floor_drift(num_seismic, floors)
    # Define the system
    kwargs = {"floors": floors}
    system_mtx, md_mtx = _system_matrices(acc_sensor, num_modes, kwargs=kwargs)
    Q_zeta = np.eye(num_modes * 2) * dkf_params[0]
    R = np.eye(len(acc_sensor) + len(floors)) * dkf_params[1]
    Q_p = np.eye(1) * dkf_params[2]
    dkf_params = {"Q_zeta": Q_zeta, "R": R, "Q_p": Q_p}
    disp_list = []
    velo_list = []
    for i in range(num_seismic):
        steps = acc_list[i].shape[0]
        disp_pred = np.zeros((steps, num_modes))
        velo_pred = np.zeros((steps, num_modes))
        obs_data = np.vstack(
            (
                floor_drift[i].cpu().numpy().T,
                acc_list[i].cpu().numpy().T,
            )
        )
        x_mtx = _akf(obs_data, system_mtx, dkf_params)

        disp_mtx = md_mtx @ x_mtx[:num_modes, :]
        velo_mtx = md_mtx @ x_mtx[num_modes : num_modes * 2, :]
        disp_pred[:, :] = disp_mtx.T
        velo_pred[:, :] = velo_mtx.T
        disp_list.append(disp_pred)
        velo_list.append(velo_pred)
    return disp_list, velo_list


def tr_training(
    RNN4ststate,
    acc_tensor,
    floor_train,
    floor_test,
    measured_drift_train,
    measured_drift_test,
    lr,
    epochs,
    unfrozen_params,
    output,
    num,
    save_path,
):
    loss_fun = torch.nn.MSELoss(reduction="mean")

    for j, param in enumerate(RNN4ststate.parameters()):
        param.requires_grad = False
        if j in unfrozen_params:
            param.requires_grad = True
    optimizer = torch.optim.Adam(RNN4ststate.parameters(), lr=lr)
    loss_history = []
    # test the prediction
    _, state_list = generate_seismic_response([0, 1, 2, 3, 4], 4, output)
    if RNN4ststate.bidirectional:
        h0 = torch.zeros(2, 1, 30).to(device)
    else:
        h0 = torch.zeros(1, 1, 30).to(device)
    for epoch in range(epochs):
        if epoch == 0:
            state_pred, _ = RNN4ststate(acc_tensor, h0)
            state_pred = state_pred.squeeze()
            state_pred = state_pred.cpu().detach().numpy()
            plt.plot(
                state_list[num].cpu().numpy()[:, 6],
                label="Ground truth",
                color="k",
                linewidth=0.8,
            )
            plt.plot(
                state_pred[:, 6],
                label="Prediction",
                linestyle="--",
                color="b",
                linewidth=0.8,
            )
            plt.legend()
        optimizer.zero_grad()
        drift_pred_train = floor_drift_pred(RNN4ststate, acc_tensor, floor_train)
        RNN4ststate.train()
        loss = loss_fun(drift_pred_train, measured_drift_train)
        loss.backward()
        optimizer.step()
        drift_pred_test = floor_drift_pred(RNN4ststate, acc_tensor, floor_test)
        test_loss = loss_fun(drift_pred_test, measured_drift_test)
        loss_history.append([loss.item(), test_loss.item()])
        if epoch > 3 and loss_history[-1][1] < loss_history[-2][1]:
            with open(save_path, "wb") as f:
                torch.save(RNN4ststate.state_dict(), f)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}, Training Loss: {loss.item()}, Test Loss: {test_loss.item()}"
            )
    with open(save_path, "rb") as f:
        RNN4ststate.load_state_dict(torch.load(f))
    state_pred, _ = RNN4ststate(acc_tensor, h0)
    state_pred = state_pred.squeeze()
    state_pred = state_pred.cpu().detach().numpy()
    plt.plot(
        state_pred[:, 6],
        label="Prediction",
        linestyle="-.",
        color="r",
        linewidth=0.8,
    )
    plt.legend()
    plt.show()
    return loss_history


def tr_birnn(output="all"):
    # transfer learning of birnn
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 4
    floors_train = [
        [0, 1],
        [1, 2],
        [2, 3],
    ]
    floors_test = [[3, 4]]
    acc_list, state_list = generate_seismic_response(acc_sensor, num_seismic)
    drift_train_list = generate_floor_drift(num_seismic, floors_train)
    drift_test_list = generate_floor_drift(num_seismic, floors_test)
    lr = 1e-5
    epochs = 10000
    output_size = 26 if output == "all" else 13
    for i in range(num_seismic):
        RNN4ststate = Rnn(
            input_size=len(acc_sensor),
            hidden_size=30,
            output_size=output_size,
            num_layers=1,
            bidirectional=True,
        )
        with open("./dataset/sts/birnn.pth", "rb") as f:
            RNN4ststate.load_state_dict(torch.load(f))
        acc_tensor = acc_list[i].unsqueeze(0)
        measured_drift_train = drift_train_list[i]
        measured_drift_test = drift_test_list[i]
        loss_history = tr_training(
            RNN4ststate,
            acc_tensor,
            floors_train,
            floors_test,
            measured_drift_train,
            measured_drift_test,
            lr,
            epochs,
            unfrozen_params=[0, 1, 2, 3],
            output=output,
            num=i,
            save_path="./dataset/sts/tr_birnn" + format(i, "03") + ".pth",
        )
        with open("./dataset/sts/tr_birnn" + format(i, "03") + ".pkl", "wb") as f:
            pickle.dump(loss_history, f)
        loss_history = np.array(loss_history)
        plt.plot(loss_history[:, 0], label="Training Loss")
        plt.plot(loss_history[:, 1], label="Test Loss")
        plt.yscale("log")
        plt.legend()
        plt.show()


def tr_rnn():
    # transfer learning for rnn
    acc_sensor = [0, 1, 2, 3, 4]
    num_seismic = 4
    floors_train = [
        [0, 1],
        [1, 2],
        [2, 3],
    ]
    floors_test = [[3, 4]]
    acc_list, _ = generate_seismic_response(acc_sensor, num_seismic)
    drift_train_list = generate_floor_drift(num_seismic, floors_train)
    drift_test_list = generate_floor_drift(num_seismic, floors_test)
    lr = 1e-5
    epochs = 10000
    for i in range(num_seismic):
        RNN4ststate = Rnn(
            input_size=len(acc_sensor),
            hidden_size=30,
            output_size=26,
            num_layers=1,
            bidirectional=False,
        )
        with open("./dataset/sts/rnn.pth", "rb") as f:
            RNN4ststate.load_state_dict(torch.load(f))
        acc_tensor = acc_list[i].unsqueeze(0)
        measured_drift_train = drift_train_list[i]
        measured_drift_test = drift_test_list[i]
        loss_history = tr_training(
            RNN4ststate,
            acc_tensor,
            floors_train,
            floors_test,
            measured_drift_train,
            measured_drift_test,
            lr,
            epochs,
            unfrozen_params=[0, 1],
            output="all",
            num=i,
            save_path="./dataset/sts/tr_rnn" + format(i, "03") + ".pth",
        )
        with open("./dataset/sts/tr_rnn" + format(i, "03") + ".pkl", "wb") as f:
            pickle.dump(loss_history, f)
        loss_history = np.array(loss_history)
        plt.plot(loss_history[:, 0], label="Training Loss")
        plt.plot(loss_history[:, 1], label="Test Loss")
        plt.yscale("log")
        plt.legend()
        plt.show()
