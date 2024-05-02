import numpy as np
import pickle
from systems import ContinuousBeam01
from excitations import PSDExcitationGenerator, BandPassPSD
import matplotlib.pyplot as plt
from models import Rnn02
import torch
import time
import scipy.io


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def random_vibration(num=30):
    for i in range(num):
        print(f"Generating solution {i}...")
        start_time = time.time()
        psd = BandPassPSD(a_v=0.04, f_1=10.0, f_2=410.0)
        force = PSDExcitationGenerator(psd, tmax=5, fmax=2000)
        # force.plot()
        print("Force" + " generated.")
        force = force()
        sampling_freq = 10000
        samping_period = 5.0
        cb = ContinuousBeam01(
            t_eval=np.linspace(
                0,
                samping_period,
                int(sampling_freq * samping_period) + 1,
            ),
            f_t=[force],
        )
        full_data = cb.run()
        solution = {}
        solution["displacement"] = full_data["displacement"].T
        solution["acceleration"] = full_data["acceleration"].T
        solution["velocity"] = full_data["velocity"].T
        solution["force"] = full_data["force"].T
        solution["time"] = full_data["time"]
        file_name = f"./dataset/csb/solution" + format(i, "03") + ".pkl"
        with open(file_name, "wb") as f:
            pickle.dump(solution, f)
        print("File " + file_name + " saved.")
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.2f} s")


def plot_solution():
    with open("./dataset/csb/solution000.pkl", "rb") as f:
        solution = pickle.load(f)
    print(solution["time"])
    plt.plot(solution["time"], solution["displacement"][:, 0] * 1000)
    plt.plot(solution["time"], solution["displacement"][:, 36] * 1000)
    plt.show()
    plt.plot(solution["time"], solution["acceleration"][:, 22] / 100)
    plt.plot(solution["time"], solution["acceleration"][:, 44] / 100)
    plt.show()
    plt.plot(solution["time"], solution["velocity"][:, 3])
    plt.plot(solution["time"], solution["velocity"][:, 36])
    plt.show()
    plt.plot(solution["time"], solution["force"][:])
    plt.show()


def modal_analysis():
    from pyoma2.algorithm import FSDD_algo
    from pyoma2.OMA import SingleSetup

    data_path = "./dataset/csb/exp_1.mat"
    save_path = "./dataset/csb/"
    acc_tensor = _measured_acc(filename=data_path)
    acc_mtx = acc_tensor.detach().cpu().numpy()
    Pali_ss = SingleSetup(acc_mtx, fs=5000)
    fig, ax = Pali_ss.plot_data()
    fig, ax = Pali_ss.plot_ch_info(ch_idx=[-1])

    fsdd = FSDD_algo(name="FSDD", nxseg=2000, method_SD="per", pov=0.5)
    Pali_ss.add_algorithms(fsdd)
    Pali_ss.run_by_name("FSDD")
    fsdd.plot_CMIF(freqlim=(0, 500))
    plt.show()
    Pali_ss.MPE("FSDD", sel_freq=[63.3, 93.1], MAClim=0.95)
    ms_array = np.real(Pali_ss["FSDD"].result.Phi)
    nf_array = Pali_ss["FSDD"].result.Fn
    dp_array = Pali_ss["FSDD"].result.Xi
    print(1 / nf_array)
    print(dp_array)
    if save_path is not None:
        file_name = save_path + "modal_analysis.pkl"
        with open(file_name, "wb") as f:
            pickle.dump({"ms": ms_array, "nf": nf_array, "dp": dp_array}, f)
        print("File " + file_name + " saved.")


def modal_properties(i=5):
    cb = ContinuousBeam01(
        t_eval=np.linspace(0, 10, 10001),
        f_t=None,
    )
    u, v = cb.freqs_modes()
    v = v[::2, :]
    print(u)
    plt.plot(v[:, i])
    plt.show()


def training_test_data(acc_sensor, num_train_files, num_test_files):
    for i in range(num_train_files):
        filename = f"./dataset/csb/solution" + format(i, "03") + ".pkl"
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
                    solution["displacement"] * 1000,
                    solution["velocity"] * 10,
                )
            ),
            dtype=torch.float32,
        ).to(device)
        acc_train[i, :, :] = torch.tensor(
            solution["acceleration"][:, acc_sensor] / 100, dtype=torch.float32
        ).to(device)
    for i in range(num_test_files):
        filename = (
            f"./dataset/csb/solution" + format(i + num_train_files, "03") + ".pkl"
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
                    solution["displacement"] * 1000,
                    solution["velocity"] * 10,
                )
            ),
            dtype=torch.float32,
        ).to(device)
        acc_test[i, :, :] = torch.tensor(
            solution["acceleration"][:, acc_sensor] / 100, dtype=torch.float32
        ).to(device)
    return state_train, acc_train, state_test, acc_test


def _rnn(acc_sensor, num_train_files, num_test_files, epochs, lr, weight_decay=0.0):
    state_train, acc_train, state_test, acc_test = training_test_data(
        acc_sensor, num_train_files, num_test_files
    )
    train_set = {"X": acc_train, "Y": state_train}
    test_set = {"X": acc_test, "Y": state_test}
    print(f"Train set: {state_train.shape}, {acc_train.shape}")
    print(f"Test set: {state_test.shape}, {acc_test.shape}")

    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=8,
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


def build_rnn():
    acc_sensor = [24, 44, 64, 98]
    epochs = 20000
    lr = 1e-4
    train_loss_list, test_loss_list = _rnn(acc_sensor, 20, 10, epochs, lr)
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.legend()
    plt.show()


def rnn_pred(path="./dataset/csb/rnn.pth"):
    acc_sensor = [24, 44, 64, 98]
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=8,
        num_layers=1,
        output_size=256,
        bidirectional=False,
    )
    rnn.load_state_dict(torch.load(path))
    rnn.to(device)
    # load the experimental data in .mat format
    acc_tensor = _measured_acc(filename=f"./dataset/csb/exp_3.mat")
    disp_tensor = _measured_disp(filename=f"./dataset/csb/exp_3.mat")
    print(acc_tensor.shape)
    disp = disp_tensor.detach().cpu().numpy()
    train_h0 = torch.zeros(1, rnn.hidden_size, dtype=torch.float32).to(device)
    state_pred, _ = rnn(acc_tensor, train_h0)
    state_pred = state_pred.detach().cpu().numpy()
    print(state_pred.shape)
    plt.plot(state_pred[:, 34])
    plt.plot(disp)
    plt.show()
    plt.plot(state_pred[:, 34], disp, "o")
    plt.show()


def _comp_strain_from_nodal_disp(nodal_disp, loc_fbg):
    # compute strain from nodal displacement
    # nodal_disp: (nt, n_dof), torch tensor
    # loc_fbg: list of float, location of FBG sensors
    num_fbg = len(loc_fbg)
    strain = torch.zeros(nodal_disp.shape[0], num_fbg).to(device)
    y = 0.0025
    L = 0.02
    for i, loc in enumerate(loc_fbg):
        ele_num = int(loc / L)
        dofs = [2 * ele_num, 2 * ele_num + 1, 2 * ele_num + 2, 2 * ele_num + 3]
        disp = nodal_disp[:, dofs] / 1000
        x = loc - ele_num * L
        B_mtx = (
            y
            / L
            * torch.tensor(
                [
                    [6 / L - 12 * x / L**2],
                    [4 - 6 * x / L],
                    [-6 / L + 12 * x / L**2],
                    [2 - 6 * x / L],
                ],
                dtype=torch.float32,
            ).to(device)
        )
        strain[:, i] = (disp @ B_mtx).squeeze() * 1e6
    return strain


def _measured_strain(filename=f"./dataset/csb/exp_1.mat"):
    k = 0.78
    # measured strain data for training
    exp_data = scipy.io.loadmat(filename)
    fbg1_ini = exp_data["fbg1_ini"]
    fbg2_ini = exp_data["fbg2_ini"]
    fbg3_ini = exp_data["fbg3_ini"]
    fbg4_ini = exp_data["fbg4_ini"]
    strain1 = (exp_data["fbg1"][::5] - fbg1_ini) / (fbg1_ini * k)
    strain2 = (exp_data["fbg2"][::5] - fbg2_ini) / (fbg2_ini * k)
    strain3 = (exp_data["fbg3"][::5] - fbg3_ini) / (fbg3_ini * k)
    strain4 = (exp_data["fbg4"][::5] - fbg4_ini) / (fbg4_ini * k)
    strain = np.hstack((strain1, strain2, strain3, strain4))
    strain = torch.tensor(strain, dtype=torch.float32).to(device) * 1e6
    return strain


def _measured_acc(filename=f"./dataset/csb/exp_1.mat"):
    exp_data = scipy.io.loadmat(filename)
    # print(exp_data.keys())
    acc1 = exp_data["acc1"][::5] * 9.8 / 100
    acc2 = exp_data["acc2"][::5] * 9.8 / 100
    acc3 = exp_data["acc3"][::5] * 9.8 / 100
    acc4 = exp_data["acc4"][::5] * 9.8 / 100
    acc_tensor = torch.tensor(
        -np.hstack((acc4, acc3, acc2, acc1)), dtype=torch.float32
    ).to(device)
    return acc_tensor


def _measured_disp(filename=f"./dataset/csb/exp_1.mat"):
    # the measured displacement data, unit: mm
    exp_data = scipy.io.loadmat(filename)
    disp = exp_data["disp1"][::5]
    disp = torch.tensor(disp, dtype=torch.float32).to(device) * 1000
    return disp


def tr_training(
    RNN4state,
    acc_tensor,
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
        filter(lambda p: p.requires_grad, RNN4state.parameters()), lr=lr
    )
    train_loss_list = []
    test_loss_list = []
    if RNN4state.bidirectional:
        h0 = torch.zeros(2, 1, RNN4state.hidden_size, dtype=torch.float32).to(device)
    else:
        h0 = torch.zeros(1, RNN4state.hidden_size, dtype=torch.float32).to(device)
    for i in range(epochs):
        state_pred, _ = RNN4state(acc_tensor, h0)
        disp_pred = state_pred[:, :128]
        strain_train_pred = _comp_strain_from_nodal_disp(disp_pred, loc_fbg_train)
        optimizer.zero_grad()
        RNN4state.train()
        loss = loss_fun(strain_train_pred, measured_strain_train)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        with torch.no_grad():
            state_pred_test, _ = RNN4state(acc_tensor, h0)
            disp_pred_test = state_pred_test[:, :128]
            strain_test_pred = _comp_strain_from_nodal_disp(
                disp_pred_test, loc_fbg_test
            )
            test_loss = loss_fun(strain_test_pred, measured_strain_test)
            test_loss_list.append(test_loss.item())
        torch.save(RNN4state.state_dict(), save_path)
        if i % 200 == 0:
            print(
                f"Epoch {i}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}"
            )
            plt.plot(
                measured_strain_test[:, 0].detach().cpu().numpy(), label="measured"
            )
            plt.plot(strain_test_pred[:, 0].detach().cpu().numpy(), label="predicted")
            plt.legend()
            plt.show()
            rnn_pred(path=save_path)
    return train_loss_list, test_loss_list


def tr_rnn():
    acc_sensor = [24, 44, 64, 98]
    # transfer learning of recurrent neural networks
    rnn = Rnn02(
        input_size=len(acc_sensor),
        hidden_size=8,
        num_layers=1,
        output_size=256,
        bidirectional=False,
    )
    rnn.load_state_dict(torch.load(f"./dataset/csb/rnn.pth"))
    rnn.to(device)
    unfrozen_params = [0, 1]
    lr = 1e-5
    epochs = 2000
    measured_strain = _measured_strain(f"./dataset/csb/exp_3.mat")
    measured_strain_train = measured_strain[:, [0, 2, 3]]
    measured_strain_test = measured_strain[:, [1]]
    loc_fbg_train = [0.3, 0.64, 1.0]
    loc_fbg_test = [0.5]
    acc_tensor = _measured_acc(f"./dataset/csb/exp_3.mat")
    train_loss_list, test_loss_list = tr_training(
        rnn,
        acc_tensor,
        measured_strain_train,
        measured_strain_test,
        loc_fbg_train,
        loc_fbg_test,
        lr,
        epochs,
        unfrozen_params,
        "./dataset/csb/tr_rnn.pth",
    )
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.legend()
    plt.show()
