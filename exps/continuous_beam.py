import numpy as np
import pickle
from systems import ContinuousBeam01
from excitations import PSDExcitationGenerator, BandPassPSD
import matplotlib.pyplot as plt
from models import Rnn02
import torch
import time
import scipy.io
from scipy import signal
import sdypy as sd
from scipy.optimize import minimize
import types
import os

np.random.seed(0)
torch.random.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def _measured_strain(filename=f"./dataset/csb/exp_1.mat", compress_ratio=1):
    k = 0.78
    # measured strain data for training
    exp_data = scipy.io.loadmat(filename)
    fbg1_ini = exp_data["fbg1_ini"]
    fbg2_ini = exp_data["fbg2_ini"]
    fbg3_ini = exp_data["fbg3_ini"]
    strain1 = (exp_data["fbg1"][::compress_ratio] - fbg1_ini) / (fbg1_ini * k)
    strain2 = (exp_data["fbg2"][::compress_ratio] - fbg2_ini) / (fbg2_ini * k)
    strain3 = (exp_data["fbg3"][::compress_ratio] - fbg3_ini) / (fbg3_ini * k)
    strain = np.hstack((strain1, strain2, strain3))
    print(fbg1_ini, fbg2_ini, fbg3_ini)
    strain = torch.tensor(strain, dtype=torch.float32).to(device) * 1e6
    return strain


def _measured_acc(
    filename=f"./dataset/csb/exp_1.mat", acc_scale=0.01, compress_ratio=1
):
    exp_data = scipy.io.loadmat(filename)
    acc1 = -exp_data["acc1"][::compress_ratio] * 9.8 * acc_scale
    acc2 = -exp_data["acc2"][::compress_ratio] * 9.8 * acc_scale
    acc3 = -exp_data["acc3"][::compress_ratio] * 9.8 * acc_scale
    acc_tensor = torch.tensor(np.hstack((acc1, acc2, acc3)), dtype=torch.float32).to(
        device
    )
    return acc_tensor


def _measured_disp(filename=f"./dataset/csb/exp_1.mat", disp_scale=1, compress_ratio=1):
    # the measured displacement data, unit: mm
    exp_data = scipy.io.loadmat(filename)
    disp = exp_data["disp1"][::compress_ratio]
    disp = torch.tensor(disp, dtype=torch.float32).to(device) * disp_scale
    return disp


def _measured_force(filename=f"./dataset/csb/exp_1.mat"):
    # the measured force data, unit: N
    exp_data = scipy.io.loadmat(filename)
    force = exp_data["force1"][::5]
    force = torch.tensor(force, dtype=torch.float32).to(device)
    return force


def ema():
    save_path = "./dataset/csb/ema.pkl"
    data_path = "./dataset/csb/noise_for_ema_2.mat"
    exp_data = scipy.io.loadmat(data_path)
    acc1 = -exp_data["Data1_AI_1_AI_1"][::5, 0].reshape(-1, 1) * 9.8
    acc2 = -exp_data["Data1_AI_2_AI_2"][::5, 0].reshape(-1, 1) * 9.8
    acc3 = -exp_data["Data1_AI_3_AI_3"][::5, 0].reshape(-1, 1) * 9.8
    acc_mtx = np.hstack((acc1, acc2, acc3))
    acc_mtx = np.array([acc_mtx.T], dtype=np.float32)
    print(acc_mtx.shape)
    force = exp_data["Data1_AI_4_AI_4"][::5, 0].reshape(-1, 1)
    force_mtx = np.array([force.T], dtype=np.float32)
    frf_obj = sd.FRF.FRF(
        1000,
        force_mtx,
        acc_mtx,
        window="hann",
        fft_len=5000,
        nperseg=5000,
    )
    df = 0.2
    freq = np.arange(0, 500 + 0.2, df)
    frf = frf_obj.get_FRF(type="default", form="accelerance")
    frf = frf.squeeze()
    plt.plot(freq, np.abs(frf.T))
    plt.legend(["acc1", "acc2", "acc3"])
    plt.yscale("log")
    plt.show()
    # save frf data from 10 hertz to 100 hertz
    idx_low = 50
    idx_high = 500
    print(freq[idx_low], freq[idx_high])
    frf = frf[:, idx_low:idx_high].T
    with open(save_path, "wb") as f:
        pickle.dump(frf, f)
    print(f"EMA data saved to {save_path}")


def compute_frf(
    k_theta_1_factor,
    k_s_factor,
    k_theta_2_factor,
    k_theta_3_factor,
    rho_factor,
    E_factor,
    damp,
    resized=False,
):
    k_theta_1 = 1e3 * k_theta_1_factor
    k_s = 1e5 * k_s_factor
    k_theta_2 = 1e1 * k_theta_2_factor
    k_theta_3 = 1e3 * k_theta_3_factor
    material_properties = {
        "elastic_modulus": 68.5e9 * E_factor,
        "density": 2.7e3 * rho_factor,
        "left_support_rotational_stiffness": k_theta_1,
        "mid_support_rotational_stiffness": k_theta_2,
        "right_support_rotational_stiffness": k_theta_3,
        "mid_support_translational_stiffness": k_s,
    }
    cb = ContinuousBeam01(
        t_eval=np.linspace(0, 10, 1001),
        f_t=[1],
        material_properties=material_properties,
        damping_params=(0, 1, damp),
    )
    frf_mtx = cb.frf(resized)
    return frf_mtx


def loss_func(params, frf_data):
    # result = types.SimpleNamespace()
    # result.x = params
    # idx = [*range(50, 100)] + [*range(300, 350)]
    # frf_data = frf_data[idx, :]
    frf_mtx = compute_frf(*params)
    phase_data = np.unwrap(np.abs(np.angle(frf_data)))
    phase_mtx = np.unwrap(np.abs(np.angle(frf_mtx)))
    amp_data = np.abs(frf_data)
    amp_mtx = np.abs(frf_mtx)
    loss = np.sum(((np.log10(amp_data) - np.log10(amp_mtx))) ** 2) + np.sum(
        ((phase_data - phase_mtx)) ** 2
    )
    loss = loss / (frf_data.shape[0] * frf_data.shape[1])
    # compare_frf(result, frf_data, resized=True)
    return loss


def compare_frf(result, frf_data, resized=False):
    (
        k_theta_1_factor,
        k_s_factor,
        k_theta_2_factor,
        k_theta_3_factor,
        rho_factor,
        E_factor,
        damp,
    ) = result.x
    frf_mtx = compute_frf(
        k_theta_1_factor,
        k_s_factor,
        k_theta_2_factor,
        k_theta_3_factor,
        rho_factor,
        E_factor,
        damp,
        resized,
    )
    freq = np.linspace(10, 100, 450)
    if resized:
        idx = [*range(50, 100)] + [*range(300, 350)]
        freq = freq[idx]
    _, ax = plt.subplots(3, 1, figsize=(8, 6))
    for i in range(3):
        ax[i].plot(freq, np.abs(frf_mtx[:, i]), linestyle="--", color="red")
        ax[i].plot(freq, np.abs(frf_data[:, i]), color="blue")
        ax[i].set_yscale("log")
    plt.show()

    _, ax = plt.subplots(3, 1, figsize=(8, 6))
    for i in range(3):
        ax[i].plot(
            freq,
            np.unwrap(np.abs(np.angle(frf_mtx[:, i]))) / np.pi,
            color="red",
            linestyle="--",
        )
        ax[i].plot(
            freq,
            np.unwrap(np.abs(np.angle(frf_data[:, i]))) / np.pi,
            color="blue",
        )

    plt.legend(["Model", "Data"])
    plt.show()


# def plot_mode_shape(result, order=2):
#     (
#         k_theta_1_factor,
#         k_s_factor,
#         k_theta_2_factor,
#         k_theta_3_factor,
#         rho_factor,
#         E_factor,
#         damp,
#     ) = result.x
#     k_theta_1 = 1e3 * k_theta_1_factor
#     k_s = 1e5 * k_s_factor
#     k_theta_2 = 1e2 * k_theta_2_factor
#     k_theta_3 = 1e3 * k_theta_3_factor
#     material_properties = {
#         "elastic_modulus": 68.5e9 * E_factor,
#         "density": 2.7e3 * rho_factor,
#         "left_support_rotational_stiffness": k_theta_1,
#         "mid_support_rotational_stiffness": k_theta_2,
#         "right_support_rotational_stiffness": k_theta_3,
#         "mid_support_translational_stiffness": k_s,
#     }
#     cb = ContinuousBeam01(
#         t_eval=np.linspace(0, 10, 1001),
#         f_t=[1],
#         material_properties=material_properties,
#         damping_params=(0, 1, damp),
#     )
#     ns, ms = cb.freqs_modes()
#     print(f"Natural frequencies: {ns}")
#     plt.plot(ms[::2, order - 1])
#     plt.text(3, 0.9, f"{ms[0, order - 1]:.2f}")
#     plt.show()


def initial_comparsion():
    import types

    result = types.SimpleNamespace()
    result.x = [1.5, 1.0, 0.0, 0.5, 1.2, 0.9, 0.012]
    with open("./dataset/csb/ema.pkl", "rb") as f:
        frf_data = pickle.load(f)
    compare_frf(result, frf_data, resized=False)
    # plot_mode_shape(result, 1)
    # plot_mode_shape(result, 2)


def model_updating():
    with open("./dataset/csb/ema.pkl", "rb") as f:
        frf_data = pickle.load(f)
    # Initial guess for parameters
    initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.008]
    # Minimize the loss function
    result = minimize(
        loss_func,  # function to minimize
        x0=initial_guess,  # initial guess
        args=(frf_data,),  # additional arguments passed to loss_func
        method="L-BFGS-B",  # optimization method
        options={"disp": True},
        bounds=[
            (0, 10.0),
            (0, 10.0),
            (0, 10.0),
            (0, 10.0),
            (0.5, 1.5),
            (0.5, 1.5),
            (0.003, 0.03),
        ],
    )
    compare_frf(result, frf_data)
    print(result.x)
    save_path = "./dataset/csb/model_updating.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


def random_vibration(num=8):
    result_path = "./dataset/csb/model_updating.pkl"
    with open(result_path, "rb") as f:
        result = pickle.load(f)
    (
        k_theta_1_factor,
        k_s_factor,
        k_theta_2_factor,
        k_theta_3_factor,
        rho_factor,
        E_factor,
        damp,
    ) = result.x
    print(f"Model updating result: {result.x}")
    for i in range(num):
        print(f"Generating solution {i}...")
        start_time = time.time()
        psd = BandPassPSD(a_v=1.0, f_1=10.0, f_2=410.0)
        force = PSDExcitationGenerator(
            psd, tmax=11, fmax=1000, normalize=True, normalize_factor=0.2
        )
        # force.plot()
        print("Force" + " generated.")
        force = force()
        sampling_freq = 10000
        samping_period = 10.0
        k_theta_1 = 1e3 * k_theta_1_factor
        k_s = 1e5 * k_s_factor
        k_theta_2 = 1e1 * k_theta_2_factor
        k_theta_3 = 1e3 * k_theta_3_factor
        material_properties = {
            "elastic_modulus": 68.5e9 * E_factor,
            "density": 2.7e3 * rho_factor,
            "left_support_rotational_stiffness": k_theta_1,
            "mid_support_translational_stiffness": k_s,
            "mid_support_rotational_stiffness": k_theta_2,
            "right_support_rotational_stiffness": k_theta_3,
        }

        cb = ContinuousBeam01(
            t_eval=np.linspace(
                0,
                samping_period,
                int(sampling_freq * samping_period) + 1,
            ),
            f_t=[force],
            material_properties=material_properties,
            damping_params=(0, 1, damp),
        )
        full_data = cb.run()
        solution = {}
        solution["displacement"] = full_data["displacement"].T
        solution["acceleration"] = full_data["acceleration"].T
        solution["velocity"] = full_data["velocity"].T
        solution["force"] = full_data["force"].T
        solution["time"] = full_data["time"]
        newpath = r"./dataset/csb/training_test_data"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        file_name = (
            f"./dataset/csb/training_test_data/solution" + format(i, "03") + ".pkl"
        )
        with open(file_name, "wb") as f:
            pickle.dump(solution, f)
        print("File " + file_name + " saved.")
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.2f} s")


def plot_solution():
    with open("./dataset/csb/training_test_data/solution000.pkl", "rb") as f:
        solution = pickle.load(f)
    print(solution["time"])
    plt.plot(solution["time"], solution["displacement"][:, 0])
    plt.plot(solution["time"], solution["displacement"][:, 36])
    plt.show()
    plt.plot(solution["time"], solution["displacement"][:, 1])
    plt.plot(solution["time"], solution["displacement"][:, 37])
    plt.show()
    plt.plot(solution["time"], solution["acceleration"][:, 22])
    plt.plot(solution["time"], solution["acceleration"][:, 44])
    plt.show()
    plt.plot(solution["time"], solution["velocity"][:, 3])
    plt.plot(solution["time"], solution["velocity"][:, 36])
    plt.show()
    plt.plot(solution["time"], solution["force"][:])
    plt.show()


def training_test_data(
    acc_sensor,
    num_train_files,
    num_test_files,
    noise_factor,
    disp_scale=1000,
    velo_scale=1,
    acc_scale=0.01,
    compress_ratio=1,
):
    for i in range(num_train_files):
        filename = (
            f"./dataset/csb/training_test_data/solution" + format(i, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        if i == 0:
            total_time_step = solution["time"].shape[0]
            state_train = torch.zeros(num_train_files, total_time_step, 256).to(device)
            acc_train = torch.zeros(
                num_train_files, total_time_step, len(acc_sensor)
            ).to(device)
        state_train[i, :, :] = torch.tensor(
            np.hstack(
                (
                    solution["displacement"] * disp_scale,
                    solution["velocity"] * velo_scale,
                )
            ),
            dtype=torch.float32,
        ).to(device)
        acc_train[i, :, :] = torch.tensor(
            solution["acceleration"][:, acc_sensor] * acc_scale, dtype=torch.float32
        ).to(device)
        noise = torch.randn_like(acc_train[i, :, :]) * noise_factor
        noise[0, :] = 0
        # plt.plot(acc_train[i, :, :].detach().cpu().numpy(), color="gray")
        acc_train[i, :, :] += noise
        # plt.plot(acc_train[i, :, :].detach().cpu().numpy(), color="blue")
        # plt.show()
    for i in range(num_test_files):
        filename = (
            f"./dataset/csb/training_test_data/solution"
            + format(i + num_train_files, "03")
            + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        if i == 0:
            total_time_step = solution["time"].shape[0]
            state_test = torch.zeros(num_test_files, total_time_step, 256).to(device)
            acc_test = torch.zeros(num_test_files, total_time_step, len(acc_sensor)).to(
                device
            )
        state_test[i, :, :] = torch.tensor(
            np.hstack(
                (
                    solution["displacement"] * disp_scale,
                    solution["velocity"] * velo_scale,
                )
            ),
            dtype=torch.float32,
        ).to(device)
        acc_test[i, :, :] = torch.tensor(
            solution["acceleration"][:, acc_sensor] * acc_scale, dtype=torch.float32
        ).to(device)
        noise = torch.randn_like(acc_test[i, :, :]) * noise_factor
        noise[0, :] = 0
        acc_test[i, :, :] += noise
    return (
        state_train[:, ::compress_ratio, :],
        acc_train[:, ::compress_ratio, :],
        state_test[:, ::compress_ratio, :],
        acc_test[:, ::compress_ratio, :],
    )


def _rnn(
    acc_sensor,
    num_train_files,
    num_test_files,
    epochs,
    lr,
    noise_factor,
    compress_ratio=1,
    weight_decay=0.0,
):
    state_train, acc_train, state_test, acc_test = training_test_data(
        acc_sensor,
        num_train_files,
        num_test_files,
        noise_factor,
        compress_ratio=compress_ratio,
    )
    train_set = {"X": acc_train, "Y": state_train}
    test_set = {"X": acc_test, "Y": state_test}
    print(f"Train set: {state_train.shape}, {acc_train.shape}")
    print(f"Test set: {state_test.shape}, {acc_test.shape}")

    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=36,
        num_layers=1,
        output_size=256,
        bidirectional=False,
    )
    train_h0 = torch.zeros(1, num_train_files, rnn.hidden_size, dtype=torch.float32).to(
        device
    )
    test_h0 = torch.zeros(1, num_test_files, rnn.hidden_size, dtype=torch.float32).to(
        device
    )
    model_save_path = f"./dataset/csb/rnn.pth"
    loss_save_path = f"./dataset/csb/rnn.pkl"
    train_loss_list, test_loss_list = rnn.train_RNN(
        train_set,
        test_set,
        train_h0,
        test_h0,
        epochs,
        lr,
        model_save_path,
        loss_save_path,
        train_msg=True,
        weight_decay=weight_decay,
    )
    return train_loss_list, test_loss_list


def _birnn(
    acc_sensor,
    num_train_files,
    num_test_files,
    epochs,
    lr,
    noise_factor,
    compress_ratio=1,
    weight_decay=0.0,
):
    state_train, acc_train, state_test, acc_test = training_test_data(
        acc_sensor,
        num_train_files,
        num_test_files,
        noise_factor,
        compress_ratio=compress_ratio,
    )
    train_set = {"X": acc_train, "Y": state_train}
    test_set = {"X": acc_test, "Y": state_test}
    print(f"Train set: {state_train.shape}, {acc_train.shape}")
    print(f"Test set: {state_test.shape}, {acc_test.shape}")
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=18,
        num_layers=1,
        output_size=256,
        bidirectional=True,
    )
    train_h0 = torch.zeros(2, num_train_files, rnn.hidden_size, dtype=torch.float32).to(
        device
    )
    test_h0 = torch.zeros(2, num_test_files, rnn.hidden_size, dtype=torch.float32).to(
        device
    )
    model_save_path = f"./dataset/csb/birnn.pth"
    loss_save_path = f"./dataset/csb/birnn.pkl"
    train_loss_list, test_loss_list = rnn.train_RNN(
        train_set,
        test_set,
        train_h0,
        test_h0,
        epochs,
        lr,
        model_save_path,
        loss_save_path,
        train_msg=True,
        weight_decay=weight_decay,
    )
    return train_loss_list, test_loss_list


def build_rnn():
    acc_sensor = [24, 44, 98]
    epochs = 15000
    lr = 3e-5
    noise_factor = 0.02
    compress_ratio = 1
    train_loss_list, test_loss_list = _rnn(
        acc_sensor, 1, 1, epochs, lr, noise_factor, compress_ratio
    )
    # plt.plot(train_loss_list, label="train loss")
    # plt.plot(test_loss_list, label="test loss")
    # plt.legend()
    # plt.show()


def build_birnn():
    acc_sensor = [24, 44, 98]
    epochs = 15000
    lr = 3e-5
    noise_factor = 0.02
    compress_ratio = 1
    train_loss_list, test_loss_list = _birnn(
        acc_sensor, 1, 1, epochs, lr, noise_factor, compress_ratio
    )
    # plt.plot(train_loss_list, label="train loss")
    # plt.plot(test_loss_list, label="test loss")
    # plt.legend()
    # plt.show()


def test_rnn():
    acc_sensor = [24, 44, 98]
    _, _, state_test, acc_test = training_test_data(
        acc_sensor, 5, 3, 0.02, compress_ratio=1
    )
    test_set = {"X": acc_test, "Y": state_test}
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=36,
        num_layers=1,
        output_size=256,
        bidirectional=False,
    )
    path = "./dataset/csb/rnn.pth"
    rnn.load_state_dict(torch.load(path))
    rnn.to(device)
    test_h0 = torch.zeros(1, 1, rnn.hidden_size, dtype=torch.float32).to(device)
    state_pred, _ = rnn(test_set["X"], test_h0)
    state_pred = state_pred.detach().cpu().numpy()
    state_test = test_set["Y"].detach().cpu().numpy()
    # plt.plot(state_pred[0, :, 34])
    # plt.plot(state_test[0, :, 34])
    # plt.show()
    # plt.plot(state_pred[0, :, 35])
    # plt.plot(state_test[0, :, 35])
    # plt.show()
    # plt.plot(acc_test[0, :, :].detach().cpu().numpy())
    # plt.show()
    return state_pred, state_test


def test_birnn():
    acc_sensor = [24, 44, 98]
    _, _, state_test, acc_test = training_test_data(
        acc_sensor, 5, 3, 0.02, compress_ratio=1
    )
    test_set = {"X": acc_test, "Y": state_test}
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=36,
        num_layers=1,
        output_size=256,
        bidirectional=True,
    )
    path = "./dataset/csb/birnn.pth"
    rnn.load_state_dict(torch.load(path))
    rnn.to(device)
    test_h0 = torch.zeros(2, 1, rnn.hidden_size, dtype=torch.float32).to(device)
    state_pred, _ = rnn(test_set["X"], test_h0)
    state_pred = state_pred.detach().cpu().numpy()
    state_test = test_set["Y"].detach().cpu().numpy()
    return state_pred, state_test


def rnn_pred(path="./dataset/csb/rnn.pth", plot_data=True):
    filtered_freq = 38
    shift = 40
    acc_sensor = [24, 44, 98]
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=36,
        num_layers=1,
        output_size=256,
        bidirectional=False,
    )
    rnn.load_state_dict(torch.load(path))
    rnn.to(device)
    compress_ratio = 1
    # load the experimental data in .mat format
    filename = f"./dataset/csb/exp_" + str(1) + ".mat"
    acc_tensor = _measured_acc(filename, compress_ratio=1)
    disp_tensor = _measured_disp(filename, compress_ratio=1)
    disp = disp_tensor.detach().cpu().numpy()
    train_h0 = torch.zeros(1, rnn.hidden_size, dtype=torch.float32).to(device)
    state_pred, _ = rnn(acc_tensor, train_h0)
    state_pred = state_pred.detach().cpu().numpy()
    disp_pred = state_pred[:, 34]
    # filter the predicted displacement using a low-pass filter
    b, a = signal.butter(2, filtered_freq, "lowpass", fs=5000 / compress_ratio)
    disp_pred_filtered = signal.filtfilt(b, a, disp_pred)
    b, a = signal.butter(8, filtered_freq, "lowpass", fs=5000 / compress_ratio)
    disp_data_filtered = signal.filtfilt(b, a, disp.reshape(-1))
    print(
        np.sum(
            np.sqrt(
                (disp_pred_filtered[:-shift].reshape(-1) - disp[shift:].reshape(-1))
                ** 2
            )
        )
    )
    if plot_data:
        plt.figure(figsize=(14, 6))
        plt.plot(disp[shift:], label="true", color="black")
        plt.plot(disp_pred_filtered[:-shift], label="predicted", color="red")
        # plt.plot(disp_data_filtered[shift:], label="true", color="black")
        plt.legend()
        # plt.xlim(0, 20000)
        plt.show()
    return disp_pred_filtered, disp


def birnn_pred(path="./dataset/csb/birnn.pth", plot_data=True):
    filtered_freq = 38
    shift = 40
    acc_sensor = [24, 44, 98]
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=18,
        num_layers=1,
        output_size=256,
        bidirectional=True,
    )
    rnn.load_state_dict(torch.load(path))
    rnn.to(device)
    compress_ratio = 1
    # load the experimental data in .mat format
    filename = f"./dataset/csb/exp_" + str(1) + ".mat"
    acc_tensor = _measured_acc(filename, compress_ratio=1)
    disp_tensor = _measured_disp(filename, compress_ratio=1)
    disp = disp_tensor.detach().cpu().numpy()
    train_h0 = torch.zeros(2, rnn.hidden_size, dtype=torch.float32).to(device)
    state_pred, _ = rnn(acc_tensor, train_h0)
    state_pred = state_pred.detach().cpu().numpy()
    disp_pred = state_pred[:, 34]
    # filter the predicted displacement using a low-pass filter
    b, a = signal.butter(3, filtered_freq, "lowpass", fs=5000 / compress_ratio)
    disp_pred_filtered = signal.filtfilt(b, a, disp_pred)
    b, a = signal.butter(8, filtered_freq, "lowpass", fs=5000 / compress_ratio)
    disp_data_filtered = signal.filtfilt(b, a, disp.reshape(-1))
    print(
        np.sum(
            np.sqrt(
                (
                    disp_pred_filtered[:-shift].reshape(-1)
                    - disp_data_filtered[shift:].reshape(-1)
                )
                ** 2
            )
        )
    )
    if plot_data:
        plt.figure(figsize=(14, 6))
        plt.plot(disp[shift:], label="true", color="black")
        plt.plot(disp_pred_filtered[:-shift], label="predicted", color="red")
        plt.legend()
        plt.xlim(0, 20000)
        plt.show()
    return disp_pred_filtered, disp_data_filtered


def _comp_strain_from_nodal_disp(nodal_disp, loc_fbg):
    # compute strain from nodal displacement
    # nodal_disp: (nt, n_dof), torch tensor
    # loc_fbg: list of float, location of FBG sensors
    num_fbg = len(loc_fbg)
    strain = torch.zeros(nodal_disp.shape[0], num_fbg).to(device)
    y = 0.0025

    for i, loc in enumerate(loc_fbg):
        ele_num = int(loc / 0.02)

        L = 0.04
        dofs = [2 * ele_num + 0, 2 * ele_num + 1, 2 * ele_num + 4, 2 * ele_num + 5]
        disp = nodal_disp[:, dofs]
        B_mtx = y * torch.tensor(
            [
                [6 / L**2],
                [4 / L],
                [-6 / L**2],
                [2 / L],
            ],
            dtype=torch.float32,
        ).to(device)
        strain[:, i] = (disp @ B_mtx).squeeze() * 1e3

    return strain


def tr_training(
    RNN4state,
    acc_train,
    acc_test,
    measured_strain_train,
    measured_strain_test,
    loc_fbg_train,
    loc_fbg_test,
    lr,
    epochs,
    unfrozen_params,
    save_path,
):
    loss_fun = torch.nn.MSELoss(reduction="mean")
    for j, param in enumerate(RNN4state.parameters()):
        param.requires_grad = False
        if j in unfrozen_params:
            param.requires_grad = True
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, RNN4state.parameters()),
        lr=lr,
    )
    train_loss_list = []
    test_loss_list = []
    if RNN4state.bidirectional:
        h0 = torch.zeros(2, RNN4state.hidden_size, dtype=torch.float32).to(device)
    else:
        h0 = torch.zeros(1, RNN4state.hidden_size, dtype=torch.float32).to(device)
    for i in range(epochs):
        state_pred, _ = RNN4state(acc_train, h0)
        # print(state_pred.shape)
        strain_train_pred = _comp_strain_from_nodal_disp(state_pred, loc_fbg_train)
        optimizer.zero_grad()
        RNN4state.train()
        loss = loss_fun(strain_train_pred, measured_strain_train)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        with torch.no_grad():
            state_pred_test, _ = RNN4state(acc_test, h0)
            strain_test_pred = _comp_strain_from_nodal_disp(
                state_pred_test, loc_fbg_test
            )
            loss1 = loss_fun(strain_train_pred[:, 0], measured_strain_train[:, 0])
            if strain_train_pred.shape[1] > 1:
                loss2 = loss_fun(strain_train_pred[:, 1], measured_strain_train[:, 1])
            test_loss = loss_fun(strain_test_pred, measured_strain_test)
            test_loss_list.append(test_loss.item())

        # fig, ax = plt.subplots(3, 1, figsize=(18, 18))
        # ax[0].plot(measured_strain_train[:, 0].detach().cpu().numpy(), color="gray")
        # ax[1].plot(measured_strain_train[:, 1].detach().cpu().numpy(), color="gray")
        # ax[2].plot(measured_strain_test[:, 0].detach().cpu().numpy(), color="gray")
        # ax[0].plot(strain_train_pred[:, 0].detach().cpu().numpy(), color="blue")
        # ax[1].plot(strain_train_pred[:, 1].detach().cpu().numpy(), color="blue")
        # ax[2].plot(strain_test_pred.detach().cpu().numpy(), color="blue")
        # plt.show()
        torch.save(RNN4state.state_dict(), save_path)
        if RNN4state.bidirectional:
            birnn_pred(path=save_path, plot_data=False)
        else:
            rnn_pred(path=save_path, plot_data=False)
        if i > 1000 and test_loss_list[-1] > test_loss_list[-2]:
            fig, ax = plt.subplots(3, 1, figsize=(18, 18))
            ax[0].plot(measured_strain_train[:, 0].detach().cpu().numpy(), color="gray")
            ax[1].plot(measured_strain_train[:, 1].detach().cpu().numpy(), color="gray")
            ax[2].plot(measured_strain_test[:, 0].detach().cpu().numpy(), color="gray")
            ax[0].plot(strain_train_pred[:, 0].detach().cpu().numpy(), color="blue")
            ax[1].plot(strain_train_pred[:, 1].detach().cpu().numpy(), color="blue")
            ax[2].plot(strain_test_pred.detach().cpu().numpy(), color="blue")
            plt.show()
            break
        if strain_train_pred.shape[1] > 1:
            print(
                f"Epoch {i}, Train Loss 1: {loss1.item()}, Train Loss 2: {loss2.item()}, Test Loss: {test_loss.item()}"
            )

        else:
            print(
                f"Epoch {i}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}"
            )
        # rnn_pred(path=save_path)

        # if i % 200 == 0:
        #     rnn_pred(path=save_path)
        # if i % 1000 == 0:
        # plt.plot(state_pred_test[286, :128:2].detach().cpu().numpy())
        # plt.show()
        # print(strain_train_pred[286, :])
        # print(strain_test_pred[286, :])
        # plt.plot(state_pred_test[:, 30].detach().cpu().numpy())
        # plt.plot(state_pred_test[:, 31].detach().cpu().numpy())
        # plt.plot(state_pred_test[:, 34].detach().cpu().numpy())
        # plt.plot(state_pred_test[:, 35].detach().cpu().numpy())
        # plt.show()
        # fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        # ax[0].plot(measured_strain_train[:, 0].detach().cpu().numpy(), color="blue")
        # ax[0].plot(
        #     measured_strain_train[:, 1].detach().cpu().numpy(), color="green"
        # )
        # ax[0].plot(measured_strain_test[:, 0].detach().cpu().numpy(), color="red")
        # ax[1].plot(strain_train_pred[:, 0].detach().cpu().numpy(), color="blue")
        # ax[1].plot(strain_train_pred[:, 1].detach().cpu().numpy(), color="green")
        # ax[1].plot(strain_test_pred.detach().cpu().numpy(), color="red")
        # plt.show()

    return train_loss_list, test_loss_list


# def data_shift(measured_strain, acc_tensor, shift):
#     start = 700
#     num_data = 5700
#     if shift == 0:
#         measured_strain_train = measured_strain[start:num_data, [1, 2]]
#         measured_strain_test = measured_strain[start:num_data, [0]]
#         loc_fbg_train = [0.64, 1.00]
#         loc_fbg_test = [0.30]
#         acc_train = acc_tensor[start:num_data, :]
#         acc_test = acc_tensor[start:num_data, :]
#     else:
#         measured_strain_train = measured_strain[start : num_data - shift, [0, 2]]
#         measured_strain_test = measured_strain[start : num_data - shift, [1]]
#         loc_fbg_train = [0.30, 1.0]
#         loc_fbg_test = [0.64]
#         acc_train = acc_tensor[start + shift : num_data, :]
#         acc_test = acc_tensor[start + shift : num_data, :]

#     return (
#         measured_strain_train,
#         measured_strain_test,
#         loc_fbg_train,
#         loc_fbg_test,
#         acc_train,
#         acc_test,
#     )


def tr_rnn():
    acc_sensor = [24, 44, 98]
    # transfer learning of recurrent neural networks
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=36,
        num_layers=1,
        output_size=256,
        bidirectional=False,
    )
    rnn.load_state_dict(torch.load(f"./dataset/csb/rnn.pth"))
    rnn.to(device)
    unfrozen_params = [0, 1]
    lr = 1e-5
    epochs = 2000
    measured_strain = _measured_strain(f"./dataset/csb/exp_1.mat", compress_ratio=1)
    acc_tensor = _measured_acc(f"./dataset/csb/exp_1.mat", compress_ratio=1)
    start = 700
    num_data = 5700
    measured_strain_train = measured_strain[start:num_data, [1, 2]]
    measured_strain_test = measured_strain[start:num_data, [0]]
    loc_fbg_train = [0.64, 1.00]
    loc_fbg_test = [0.30]
    acc_train = acc_tensor[start:num_data, :]
    acc_test = acc_tensor[start:num_data, :]

    train_loss_list, test_loss_list = tr_training(
        rnn,
        acc_train,
        acc_test,
        measured_strain_train,
        measured_strain_test,
        loc_fbg_train,
        loc_fbg_test,
        lr,
        epochs,
        unfrozen_params,
        f"./dataset/csb/tr_rnn.pth",
    )
    torch.save(
        {"train_loss_list": train_loss_list, "test_loss_list": test_loss_list},
        f"./dataset/csb/tr_rnn.pkl",
    )


def tr_birnn():
    acc_sensor = [24, 44, 98]
    # transfer learning of recurrent neural networks
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=18,
        num_layers=1,
        output_size=256,
        bidirectional=True,
    )
    rnn.load_state_dict(torch.load(f"./dataset/csb/birnn.pth"))
    rnn.to(device)
    unfrozen_params = [0, 1, 2, 3]
    lr = 3e-5
    epochs = 2000
    measured_strain = _measured_strain(f"./dataset/csb/exp_1.mat", compress_ratio=1)
    acc_tensor = _measured_acc(f"./dataset/csb/exp_1.mat", compress_ratio=1)
    start = 700
    num_data = 5700
    measured_strain_train = measured_strain[start:num_data, [1, 2]]
    measured_strain_test = measured_strain[start:num_data, [0]]
    loc_fbg_train = [0.64, 1.00]
    loc_fbg_test = [0.30]
    acc_train = acc_tensor[start:num_data, :]
    acc_test = acc_tensor[start:num_data, :]

    train_loss_list, test_loss_list = tr_training(
        rnn,
        acc_train,
        acc_test,
        measured_strain_train,
        measured_strain_test,
        loc_fbg_train,
        loc_fbg_test,
        lr,
        epochs,
        unfrozen_params,
        f"./dataset/csb/tr_birnn.pth",
    )

    torch.save(
        {"train_loss_list": train_loss_list, "test_loss_list": test_loss_list},
        f"./dataset/csb/tr_birnn.pkl",
    )
