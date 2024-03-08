from systems import ShearTypeStructure
import numpy as np
import pickle
from utils import compute_metrics, fdd, mac
from models import Rnn, Lstm
import torch
import matplotlib.pyplot as plt
from excitations import FlatNoisePSD, PSDExcitationGenerator


def seismic_response(
    num=1, method="Radau", save_path="./dataset/shear_type_structure/"
):
    # compute the seismic vibration response
    psd_func = FlatNoisePSD(a_v=0.8)
    excitation = PSDExcitationGenerator(psd_func, 30, 10)
    for i_th in range(num):
        time, acc_g = excitation.generate()
        stiff_factor = 1e2
        damp_factor = 3
        mass_vec = 1 * np.ones(13)
        stiff_vec = np.array([13, 12, 12, 12, 8, 8, 8, 8, 8, 5, 5, 5, 5]) * stiff_factor
        damp_vec = (
            np.array(
                [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.80, 0.50, 0.50, 0.50, 0.50]
            )
            * damp_factor
        )
        parametric_sts = ShearTypeStructure(
            mass_vec=mass_vec,
            stiff_vec=stiff_vec,
            damp_vec=damp_vec,
            t=time,
            acc_g=acc_g,
        )
        _ = parametric_sts.print_damping_ratio(10)
        _ = parametric_sts.print_natural_frequency(10)

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
            return solution
        else:
            return solution


def plot_response():
    solution = seismic_response(num=1, save_path=None)
    time = solution["time"]
    acc_g = solution["acc_g"]
    disp = solution["disp"]
    acc = solution["acc"]
    plt.plot(time, acc_g, label="Ground acceleration")
    plt.legend()
    plt.show()
    plt.plot(time, acc[0, :], label="Ground floor acceleration")
    plt.plot(time, acc[7, :], label="7th floor velocity")
    plt.show()
    plt.plot(time, disp[0, :], label="Ground floor displacement")
    plt.plot(time, disp[7, :], label="7th floor displacement")
    plt.show()


def modal_analysis():
    # data_path = "./dataset/bists/ambient_response.pkl"
    # save_path = "./dataset/sts/"
    # with open(data_path, "rb") as f:
    #     solution = pickle.load(f)
    # acc_mtx = solution["acc"]
    # f_lb_list = [0.3, 1.3, 2.2, 3.0, 3.8, 4.5, 5.1, 5.5, 6.4, 6.9, 7.4, 8.5]
    # f_ub_list = [0.8, 1.8, 2.7, 3.6, 4.4, 5.1, 5.5, 6.2, 6.8, 7.4, 7.8, 9.2]

    # for i in range(len(f_lb_list)):
    #     _, nf = fdd(
    #         acc_mtx, f_lb=f_lb_list[i], f_ub=f_ub_list[i], nperseg_num=2000, fs=20
    #     )
    #     ms, _ = fdd(
    #         acc_mtx, f_lb=f_lb_list[i], f_ub=f_ub_list[i], nperseg_num=400, fs=20
    #     )
    #     if i == 0:
    #         ms_array = ms.reshape(-1, 1)
    #         nf_array = np.array([nf])
    #     else:
    #         ms_array = np.hstack((ms_array, ms.reshape(-1, 1)))
    #         nf_array = np.hstack((nf_array, np.array([nf])))
    # if save_path is not None:
    #     file_name = save_path + "modal_analysis.pkl"
    #     with open(file_name, "wb") as f:
    #         pickle.dump({"ms": ms_array, "nf": nf_array}, f)
    #     print("File " + file_name + " saved.")
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
    if save_path is not None:
        file_name = save_path + "modal_analysis.pkl"
        with open(file_name, "wb") as f:
            pickle.dump({"ms": ms_array, "nf": nf_array}, f)
        print("File " + file_name + " saved.")


def model_modal_properties(params):
    stiff_factor = 1e2
    mass_vec = 1 * np.ones(13)
    stiff_vec = (
        np.array(
            [
                13 * params[0],
                12 * params[1],
                12 * params[2],
                12 * params[3],
                8 * params[4],
                8 * params[5],
                8 * params[6],
                8 * params[7],
                8 * params[8],
                5 * params[9],
                5 * params[10],
                5 * params[11],
                5 * params[12],
            ]
        )
        * stiff_factor
    )
    # damping ratio is 0, it has nothing to do with the natural frequencies and mode shapes
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
    print(res.x)
    data_path = "./dataset/sts/modal_analysis.pkl"
    model_nf, model_ms = model_modal_properties(res.x)
    model_nf = model_nf[0:num_modes]
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    ms = solution["ms"]
    nf = solution["nf"]
    print((model_nf[0:num_modes] - nf[0:num_modes]) / nf[0:num_modes])
    with open(save_path + "model_updating.pkl", "wb") as f:
        pickle.dump(
            {
                "model_nf": model_nf,
                "model_ms": model_ms,
                "nf": nf,
                "ms": ms,
                "params": res.x,
            },
            f,
        )
    return res.x


def damping_ratio():
    from scipy import interpolate

    data_path = "./dataset/bists/ambient_response.pkl"
    save_path = "./dataset/sts/"
    with open(data_path, "rb") as f:
        solution = pickle.load(f)
    acc_mtx = solution["acc"]
    freq, psd = fdd(acc_mtx, f_lb=0.3, f_ub=0.8, nperseg_num=2000, fs=20, psd=True)
    psd_func = interpolate.interp1d(freq, psd, kind="cubic")
    freq_interp = np.linspace(0.3, 0.8, 1000)
    psd_interp = psd_func(freq_interp)
    plt.plot(freq, psd, "o", label="Original PSD")
    plt.plot(freq_interp, psd_interp, label="Interpolated PSD")
    plt.yscale("log")
    plt.show()


def training_test_dataset():
    _ = seismic_response(
        num=50, method="Radau", save_path="./dataset/shear_type_structure/"
    )
    pass


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


def _rnn(
    acc_sensor,
    data_compression_ratio=10,
    num_training_files=10,
    epochs=10000,
    lr=0.0001,
    weight_decay=0.0,
):
    """
    :param acc_sensor: (list) list of accelerometer locations
    :param data_compression_ratio: (int) data compression ratio
    """
    num_test_files = 50 - num_training_files
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disp = []
    acc = []
    disp_test = []
    acc_test = []

    for i in range(num_training_files):
        filename = (
            "./dataset/shear_type_structure/solution" + format(i + 1, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        disp.append(solution["disp"][:, ::data_compression_ratio].T)
        acc.append(solution["acc"][acc_sensor, ::data_compression_ratio].T)

    for i in range(num_training_files, 50):
        filename = (
            "./dataset/shear_type_structure/solution" + format(i + 1, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        disp_test.append(solution["disp"][:, ::data_compression_ratio].T)
        acc_test.append(solution["acc"][acc_sensor, ::data_compression_ratio].T)

    disp = np.array(disp)
    disp = torch.tensor(disp, dtype=torch.float32).to(device)
    disp_test = np.array(disp_test)
    disp_test = torch.tensor(disp_test, dtype=torch.float32).to(device)
    acc = np.array(acc)
    acc = torch.tensor(acc, dtype=torch.float32).to(device)
    acc_test = np.array(acc_test)
    acc_test = torch.tensor(acc_test, dtype=torch.float32).to(device)
    train_set = {"X": acc, "Y": disp}
    test_set = {"X": acc_test, "Y": disp_test}

    RNN_model_disp = Rnn(
        input_size=len(acc_sensor),
        hidden_size=10,
        output_size=13,
        num_layers=1,
        bidirectional=True,
    )
    train_h0 = torch.zeros(2, num_training_files, 10).to(device)
    test_h0 = torch.zeros(2, num_test_files, 10).to(device)
    model_save_path = "./dataset/shear_type_structure/rnn_disp.pth"
    loss_save_path = "./dataset/shear_type_structure/rnn_disp_loss.pkl"
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


def build_rnn():
    dr = 10
    ntf = 40
    acc_sensor = [0, 4, 7, 11]
    _, _ = _rnn(
        acc_sensor,
        data_compression_ratio=dr,
        num_training_files=ntf,
        epochs=200000,
        lr=0.0001,
        weight_decay=0.0,
    )
    RNN_model_disp = Rnn(
        input_size=len(acc_sensor),
        hidden_size=10,
        output_size=13,
        num_layers=1,
        bidirectional=True,
    )
    with open("./dataset/shear_type_structure/rnn_disp.pth", "rb") as f:
        RNN_model_disp.load_state_dict(torch.load(f))

    RNN_model_disp.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_files = 50 - ntf
    test_h0 = torch.zeros(2, num_test_files, 10).to(device)
    disp_test = []
    acc_test = []
    for i in range(ntf, 50):
        filename = (
            "./dataset/shear_type_structure/solution" + format(i + 1, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        disp_test.append(solution["disp"][:, ::dr].T)
        acc_test.append(solution["acc"][acc_sensor, ::dr].T)
    disp_test = np.array(disp_test)
    acc_test = np.array(acc_test)
    acc_test = torch.tensor(acc_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        disp_pred, _ = RNN_model_disp(acc_test, test_h0)
    disp_pred = disp_pred.cpu().numpy()
    disp_test = disp_test[:, :, 11]
    disp_pred = disp_pred[:, :, 11]
    plt.plot(disp_test[0, :], label="Ground truth")
    plt.plot(disp_pred[0, :], label="Prediction")
    plt.legend()
    plt.show()
    filename = "./dataset/shear_type_structure/rnn_disp_loss.pkl"
    with open(filename, "rb") as f:
        loss = torch.load(f)
    plt.plot(loss["train_loss_list"], label="Train loss")
    plt.plot(loss["test_loss_list"], label="Test loss")
    plt.yscale("log")
    plt.legend()
    plt.show()
