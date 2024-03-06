from systems import BaseIsolatedStructure
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from models import Rnn
from excitations import FlatNoisePSD, PSDExcitationGenerator
from scipy import interpolate


def _read_smc_file(filename):
    with open(filename) as f:
        content = f.readlines()
    data = []
    counter = 0
    for x in content:
        if counter > 39:
            s = x
            numbers = [s[i : i + 10] for i in range(0, len(s), 10)]
            number = []
            for i in range(len(numbers) - 1):
                number.append(float(numbers[i]))
            data.append(number)
        counter += 1
    acc = np.array(data)
    time = np.linspace(0, 0.02 * (len(acc[:, 0]) - 1), len(acc[:, 0]))
    return time, acc[:, 0] * 1e-2


def seismic_response(num=1):
    acc_file_root_name = "./excitations/SMSIM/m7.0r10.0_00"
    acc_file_list = [
        acc_file_root_name + format(i, "03") + ".smc" for i in range(1, num + 1)
    ]

    for acc_file_i in acc_file_list:
        time, acc_g = _read_smc_file(acc_file_i)
        time = time[1000:6000]
        time = time - time[0]
        acc_g = acc_g[1000:6000] * 10
        interp_time = np.linspace(0, time[-1], 20000)
        interp_acc_g = interpolate.interp1d(
            time, acc_g, kind="quadratic", fill_value="extrapolate"
        )(interp_time)
        stiff_factor = 1e3
        damp_factor = 2
        mass_vec = 1 * np.ones(12)
        stiff_vec = np.array([12, 12, 12, 8, 8, 8, 5, 5, 5, 3, 2, 1]) * stiff_factor
        damp_vec = (
            np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.80, 0.50, 0.50, 0.50, 0.50])
            * damp_factor
        )
        parametric_bists = BaseIsolatedStructure(
            mass_super_vec=mass_vec,
            stiff_super_vec=stiff_vec,
            damp_super_vec=damp_vec,
            isolator_params={
                "m_b": 1,
                "c_b": 1.0 * damp_factor,
                "k_b": 13 * stiff_factor,
                "q": 5e-2,
                "A": 1.0,
                "beta": 0.5,
                "gamma": 0.5,
                "n": 2.5,
                "z_0": 0.0,
                "F_y": 10.0,
                "alpha": 0.7,
            },
            x_0=np.zeros(13),
            x_dot_0=np.zeros(13),
            t=interp_time,
            acc_g=interp_acc_g,
        )
        disp, velo, acc, z = parametric_bists.run(force_type="ground motion")

        solution = {
            "acc_g": interp_acc_g,
            "time": interp_time,
            "disp": disp,
            "velo": velo,
            "acc": acc,
            "z": z,
        }

        file_name = (
            "./dataset/base_isolated_structure/solution"
            + format(acc_file_list.index(acc_file_i) + 1, "03")
            + ".pkl"
        )
        with open(file_name, "wb") as f:
            pickle.dump(solution, f)
        print("File " + file_name + " saved.")
        _ = parametric_bists.print_natural_frequency(10)
    return solution


def seismic_response_sample():
    solution = seismic_response(1)
    time = solution["time"]
    acc = solution["acc"]
    disp = solution["disp"]
    acc_g = solution["acc_g"]
    z = solution["z"]
    plt.plot(time, acc_g.T, label="Ground motion")
    plt.legend()
    plt.show()
    plt.plot(time, disp[0, :].T, label="Ground floor displacement")
    plt.plot(time, disp[8, :].T + disp[0, :].T, label="8th floor displacement")
    plt.plot(time, disp[12, :].T + disp[0, :].T, label="12th floor displacement")
    plt.legend()
    plt.show()
    plt.plot(time, np.squeeze(z))
    plt.title("Base isolation system displacement")
    plt.show()
    plt.plot(time, acc[0, :].T, label="Ground floor acceleration")
    plt.plot(time, acc[8, :].T + acc[0, :].T, label="8th floor acceleration")
    plt.plot(time, acc[12, :].T + acc[0, :].T, label="12th floor acceleration")
    plt.legend()
    plt.show()


def analytical_validation():
    # compare the results with shear type structure when the nonlinearity is turned off
    # results from shear type structure is computed based on DOP853 method
    # the results from base isolated structure is computed based on Newmark-beta method
    time = np.linspace(0, 10, 10000)
    acc = np.sin(2 * np.pi * 1 * time)
    mass_vec = 2 * np.ones(2)
    stiff_vec = 10 * np.ones(2)
    damp_vec = 0.1 * np.ones(2)
    parametric_bists = BaseIsolatedStructure(
        mass_super_vec=mass_vec,
        stiff_super_vec=stiff_vec,
        damp_super_vec=damp_vec,
        isolator_params={
            "m_b": 1,
            "c_b": 0.1,
            "k_b": 10,
            "q": 4e-3,
            "A": 0,
            "beta": 0,
            "gamma": 0,
            "n": 1,
            "z_0": 0,
            "F_y": 0,
            "alpha": 1,
        },
        x_0=np.zeros(3),
        x_dot_0=np.zeros(3),
        t=time,
        acc_g=acc,
    )

    disp, velo, acc, z = parametric_bists.run()
    _ = parametric_bists.print_natural_frequency(3)
    solution = {
        "acc_g": parametric_bists.acc_g,
        "time": time,
        "disp": disp,
        "velo": velo,
        "acc": acc,
        "z": z,
    }
    return solution


def ambient_response():
    # compute the ambient vibration response
    psd_func = FlatNoisePSD(a_v=1e-3)
    excitation = PSDExcitationGenerator(psd_func, 1000, 10)
    for i in range(13):
        time, ext = excitation.generate()
        if i == 0:
            ext_all = ext
        else:
            ext_all = np.vstack((ext_all, ext))
    stiff_factor = 1e2
    damp_factor = 5
    mass_vec = 1 * np.ones(12)
    stiff_vec = np.array([12, 12, 12, 8, 8, 8, 5, 5, 5, 3, 2, 1]) * stiff_factor
    damp_vec = (
        np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.80, 0.50, 0.50, 0.50, 0.50])
        * damp_factor
    )
    parametric_bists = BaseIsolatedStructure(
        mass_super_vec=mass_vec,
        stiff_super_vec=stiff_vec,
        damp_super_vec=damp_vec,
        isolator_params={
            "m_b": 1,
            "c_b": 1.0 * damp_factor,
            "k_b": 13 * stiff_factor,
            "q": 1e-2,
            "A": 1,
            "beta": 0.5,
            "gamma": 0.5,
            "n": 2,
            "z_0": 0,
            "F_y": 10,
            "alpha": 0.7,
        },
        x_0=np.zeros(13),
        x_dot_0=np.zeros(13),
        t=time,
        ambient_excitation=ext_all,
    )
    parametric_bists.print_natural_frequency(10)
    disp, _, acc, z = parametric_bists.run(force_type="ambient excitation")
    acc_8 = acc[8, :] + acc[0, :]
    acc_3 = acc[3, :] + acc[0, :]
    acc_11 = acc[11, :] + acc[0, :]
    disp_8 = disp[8, :] + disp[0, :]
    # FFT
    N = len(acc_8)
    T = time[-1] / N
    yf8 = np.fft.fft(acc_8)
    yf3 = np.fft.fft(acc_3)
    yf11 = np.fft.fft(acc_11)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    plt.plot(ext_all[0, :].T)
    plt.title("Ambient excitation at ground floor")
    plt.show()
    plt.plot(xf, 2.0 / N * np.abs(yf8[: N // 2]))
    plt.plot(xf, 2.0 / N * np.abs(yf3[: N // 2]))
    plt.plot(xf, 2.0 / N * np.abs(yf11[: N // 2]))
    plt.yscale("log")
    plt.legend(["Floor 8", "Floor 3", "Floor 11"])
    plt.show()
    plt.plot(time, disp_8)
    plt.title("Floor 8 displacement")
    plt.show()
    plt.plot(time, acc_8)
    plt.title("Floor 8 acceleration")
    plt.show()
    plt.plot(time, np.squeeze(z))
    plt.title("z displacement")
    plt.show()
    pass


def compute_floor_drift_bists(disp_bists, drift_sensor):
    # disp_bists is a 3D array with shape (num_files, num_time_steps, num_dofs)
    # drift_sensor is a list of integers with ascending order
    num_files, num_time_steps, _ = disp_bists.shape
    drift_bists = np.zeros((num_files, num_time_steps, len(drift_sensor)))
    if drift_sensor[0] == 0:
        drift_bists[:, :, 0] = disp_bists[:, :, 0]
        disp_bists[:, :, 0] = 0
        for i in range(1, len(drift_sensor)):
            drift_bists[:, :, i] = (
                disp_bists[:, :, drift_sensor[i]]
                - disp_bists[:, :, drift_sensor[i - 1]]
            )
    else:
        disp_bists[:, :, 0] = 0
        for i in range(len(drift_sensor)):
            drift_bists[:, :, i] = (
                disp_bists[:, :, drift_sensor[i]]
                - disp_bists[:, :, drift_sensor[i - 1]]
            )
    return drift_bists


def compute_floor_acceleration_bists(acc_bists, acc_sensor):
    # acc_bists is a 3D array with shape (num_files, num_time_steps, num_dofs)
    # acc_sensor is a list of integers with ascending order
    num_files, num_time_steps, _ = acc_bists.shape
    acc_bists_new = np.zeros((num_files, num_time_steps, len(acc_sensor)))
    if acc_sensor[0] == 0:
        acc_bists_new[:, :, 0] = acc_bists[:, :, 0]
        for i in range(1, len(acc_sensor)):
            acc_bists_new[:, :, i] = acc_bists[:, :, acc_sensor[i]] + acc_bists[:, :, 0]
    else:
        for i in range(len(acc_sensor)):
            acc_bists_new[:, :, i] = acc_bists[:, :, acc_sensor[i]] + acc_bists[:, :, 0]
    return acc_bists_new


def compute_floor_disp_bists(disp_bists):
    # disp_bists is a 3D array with shape (num_files, num_time_steps, num_dofs)
    # disp_sensor is a list of integers with ascending order
    disp_bists[:, :, 1:] = disp_bists[:, :, 1:] + disp_bists[:, :, 0:1]
    return disp_bists


def compute_floor_drift_sts(disp_sts, drift_sensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # disp_sts is a 3D tensor with shape (num_files, num_time_steps, num_dofs)
    # drift_sensor is a list of integers with ascending order
    num_files, num_time_steps, _ = disp_sts.shape
    drift_sts = torch.zeros(num_files, num_time_steps, len(drift_sensor)).to(device)
    if drift_sensor[0] == 0:
        drift_sts[:, :, 0] = disp_sts[:, :, 0]
        for i in range(1, len(drift_sensor)):
            drift_sts[:, :, i] = (
                disp_sts[:, :, drift_sensor[i]] - disp_sts[:, :, drift_sensor[i - 1]]
            )
    else:
        for i in range(len(drift_sensor)):
            drift_sts[:, :, i] = (
                disp_sts[:, :, drift_sensor[i]] - disp_sts[:, :, drift_sensor[i - 1]]
            )
    return drift_sts


def lf_rnn_prediction(which=1, dof=0):
    # low fidelity recurrent neural network prediction
    # i.e., we use the pre-trained model to predict the displacement
    # which: int, is the order of the file
    # dof: int, is the degree of freedom to be ploted
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dr = 10
    acc_sensor = [0, 4, 7, 11]
    RNN_model_disp = Rnn(
        input_size=len(acc_sensor),
        hidden_size=10,
        output_size=13,
        num_layers=1,
        bidirectional=True,
    )
    with open("./dataset/shear_type_structure/rnn_disp.pth", "rb") as f:
        RNN_model_disp.load_state_dict(torch.load(f))
    h0 = torch.zeros(2, 50, 10).to(device)
    acc_test = []
    disp_test = []
    # for param in RNN_model_disp.parameters():
    #     print(type(param))
    RNN_model_disp.eval()
    for i in range(50):
        filename = (
            "./dataset/base_isolated_structure/solution" + format(i + 1, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        acc1 = solution["acc"][acc_sensor, ::dr].T
        acc1[:, 1:] += acc1[:, 0:1]
        disp1 = solution["disp"][:, ::dr].T
        disp1[:, 1:] += disp1[:, 0:1]
        acc_test.append(acc1)
        disp_test.append(disp1)
    time = solution["time"][::dr]
    acc_test = np.array(acc_test)
    disp_test = np.array(disp_test)
    acc_test = torch.tensor(acc_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        disp_pred, _ = RNN_model_disp(acc_test, h0)
    disp_pred = disp_pred.cpu().numpy()
    # plot the prediction and ground truth
    plt.plot(time, disp_test[which - 1, :, dof], label="Ground truth")
    plt.plot(time, disp_pred[which - 1, :, dof], label="Prediction")
    plt.legend()
    plt.show()


def _tr_rnn(
    acc_sensor, drift_sensor, data_compression_ratio, num_training_files, epochs, lr
):
    # transfer learning of recurrent neural networks
    # acc_sensor: list of integers, is the indices of the acceleration sensors
    # drift_sensor: list of integers, is the indices of the drift sensors
    # data_compression_ratio: int, is the ratio of the data compression
    # num_training_files: int, is the number of training files
    # epochs: int, is the number of epochs
    # lr: float, is the learning rate

    # load the pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RNN_model_disp = Rnn(
        input_size=len(acc_sensor),
        hidden_size=10,
        output_size=13,
        num_layers=1,
        bidirectional=True,
    )
    with open("./dataset/shear_type_structure/rnn_disp.pth", "rb") as f:
        RNN_model_disp.load_state_dict(torch.load(f))
    h0 = torch.zeros(2, num_training_files, 10).to(device)
    acc_train = []
    disp_train = []
    for i in range(num_training_files):
        filename = (
            "./dataset/base_isolated_structure/solution" + format(i + 1, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        acc_train.append(solution["acc"][:, ::data_compression_ratio].T)
        disp_train.append(solution["disp"][:, ::data_compression_ratio].T)
    acc_train = compute_floor_acceleration_bists(np.array(acc_train), acc_sensor)
    drift_train = compute_floor_drift_bists(np.array(disp_train), drift_sensor)
    disp_train = compute_floor_disp_bists(np.array(disp_train))
    acc_train = torch.tensor(acc_train, dtype=torch.float32).to(device)
    disp_train = torch.tensor(disp_train, dtype=torch.float32).to(device)
    drift_train = torch.tensor(drift_train, dtype=torch.float32).to(device)
    # transfer learning, freeze the pre-trained layers
    for i, param in enumerate(RNN_model_disp.parameters()):
        param.requires_grad = False
        if i == 3 or i == 4 or i == 5:
            param.requires_grad = True
        # if i ==5:
        #     param.requires_grad = True

    criterion = torch.nn.MSELoss(reduction="mean")
    # criterion = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, RNN_model_disp.parameters()), lr=lr
    )
    loss_list = []
    for epoch in range(epochs):
        RNN_model_disp.train()
        optimizer.zero_grad()
        disp_pred, _ = RNN_model_disp(acc_train, h0)
        drift_pred = compute_floor_drift_sts(disp_pred, drift_sensor)
        loss = criterion(drift_pred, drift_train)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if epoch % 100 == 0:
            RNN_model_disp.eval()
            disp_pred, _ = RNN_model_disp(acc_train, h0)
            disp_loss = criterion(disp_pred, disp_train)
            print("Epoch: ", epoch, "Drift Loss: ", loss.item())
            print("Epoch: ", epoch, "Disp Loss: ", disp_loss.item())
    plt.plot(loss_list)
    plt.show()
    return RNN_model_disp


def build_tr_rnn():
    "Transfer learning of Recurrent neural networks"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dr = 10
    ntf = 1
    acc_sensor = [0, 4, 7, 11]
    drift_sensor = [0, 2, 4, 6]
    hf_rnn = _tr_rnn(
        acc_sensor=acc_sensor,
        drift_sensor=drift_sensor,
        data_compression_ratio=dr,
        num_training_files=ntf,
        epochs=10000,
        lr=0.00001,
    )
    model_save_path = "./dataset/base_isolated_structure/hf_rnn.pth"
    torch.save(hf_rnn.state_dict(), model_save_path)
    # model_save_path = "./dataset/base_isolated_structure/hf_rnn.pth"
    # hf_rnn = Rnn(
    #     input_size=len(acc_sensor),
    #     hidden_size=10,
    #     output_size=13,
    #     num_layers=1,
    #     bidirectional=True,
    # )
    with open(model_save_path, "rb") as f:
        hf_rnn.load_state_dict(torch.load(f))
    acc_test = []
    disp_test = []
    for i in range(50):
        filename = (
            "./dataset/base_isolated_structure/solution" + format(i + 1, "03") + ".pkl"
        )
        with open(filename, "rb") as f:
            solution = pickle.load(f)
        acc1 = solution["acc"][acc_sensor, ::dr].T
        acc1[:, 1:] += acc1[:, 0:1]
        disp1 = solution["disp"][:, ::dr].T
        disp1[:, 1:] += disp1[:, 0:1]
        acc_test.append(acc1)
        disp_test.append(disp1)
    time = solution["time"][::dr]
    acc_test = np.array(acc_test)
    disp_test = np.array(disp_test)
    acc_test = torch.tensor(acc_test, dtype=torch.float32).to(device)
    h0 = torch.zeros(2, 50, 10).to(device)
    with torch.no_grad():
        disp_pred, _ = hf_rnn(acc_test, h0)
    disp_pred = disp_pred.cpu().numpy()
    # plot the prediction and ground truth
    plt.plot(time, disp_test[0, :, 0], label="Ground truth")
    plt.plot(time, disp_pred[0, :, 0], label="Prediction")
    plt.legend()
    plt.show()
    plt.plot(time, disp_test[0, :, 7], label="Ground truth")
    plt.plot(time, disp_pred[0, :, 7], label="Prediction")
    plt.legend()
    plt.show()
    plt.plot()
