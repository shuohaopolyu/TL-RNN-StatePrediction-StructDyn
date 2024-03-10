from systems import ShearTypeStructure
import numpy as np
import pickle
from utils import compute_metrics, fdd, mac
from models import Rnn, Lstm
import torch
import matplotlib.pyplot as plt
from excitations import FlatNoisePSD, PSDExcitationGenerator


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


def _training_test_data(acc_sensor, data_compression_ratio=1, num_training_files=40):
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
        # disp.append(solution["disp"][:, :].T)
        # velo.append(solution["velo"][:, :].T)
        disp = solution["disp"][:, :].T
        velo = solution["velo"][:, :].T
        state.append(np.hstack((disp, velo)))
        acc.append(solution["acc"][acc_sensor].T)
    for i in range(num_training_files, 50):
        filename = "./dataset/sts/solution" + format(i, "03") + ".pkl"
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        # disp_test.append(solution["disp"][:, ::data_compression_ratio].T)
        # velo_test.append(solution["velo"][:, ::data_compression_ratio].T)
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
    epochs=10000,
    lr=1e-5,
    weight_decay=0.0,
):
    """
    :param acc_sensor: (list) list of accelerometer locations
    :param data_compression_ratio: (int) data compression ratio
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, acc, state_test, acc_test = _training_test_data(
        acc_sensor, data_compression_ratio, num_training_files
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
    test_h0 = torch.zeros(2, 50 - num_training_files, RNN_model_disp.hidden_size).to(
        device
    )
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
    epochs=10000,
    lr=1e-5,
    weight_decay=1e-7,
):
    """
    :param acc_sensor: (list) list of accelerometer locations
    :param data_compression_ratio: (int) data compression ratio
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, acc, state_test, acc_test = _training_test_data(
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
    test_h0 = torch.zeros(1, 50 - num_training_files, RNN_model_disp.hidden_size).to(
        device
    )
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
    ntf = 40
    acc_sensor = [0, 1, 2, 3, 4]
    _, _ = _birnn(
        acc_sensor,
        data_compression_ratio=dr,
        num_training_files=ntf,
        epochs=50000,
        lr=1e-5,
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
    num_test_files = 50 - ntf
    test_h0 = torch.zeros(2, num_test_files, 30).to(device)
    _, _, state_test, acc_test = _training_test_data(acc_sensor, dr, ntf)
    with torch.no_grad():
        state_pred, _ = RNN4ststate(acc_test, test_h0)
    state_pred = state_pred.cpu().numpy()
    state_pred = state_pred[:, :, 16]
    state_test = state_test.cpu().numpy()
    state_test = state_test[:, :, 16]
    plt.plot(state_test[3, :], label="Ground truth")
    plt.plot(state_pred[3, :], label="Prediction", linestyle="--")
    plt.legend()
    plt.show()


def build_rnn():
    dr = 1
    ntf = 40
    acc_sensor = [0, 1, 2, 3, 4]
    _, _ = _rnn(
        acc_sensor,
        data_compression_ratio=dr,
        num_training_files=ntf,
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
    num_test_files = 50 - ntf
    test_h0 = torch.zeros(1, num_test_files, 30).to(device)
    _, _, state_test, acc_test = _training_test_data(acc_sensor, dr, ntf)
    with torch.no_grad():
        state_pred, _ = RNN4ststate(acc_test, test_h0)
    state_pred = state_pred.cpu().numpy()
    state_pred = state_pred[:, :, 16]
    state_test = state_test.cpu().numpy()
    state_test = state_test[:, :, 16]
    plt.plot(state_test[3, :], label="Ground truth")
    plt.plot(state_pred[3, :], label="Prediction", linestyle="--")
    plt.legend()
    plt.show()
